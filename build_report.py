"""Build the final Chinese PDF from verified experiment outputs only."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line

ROOT = Path(__file__).resolve().parent
NAMES = {"binary_bow": "Binary BoW", "word_frequency": "Word Frequency", "tfidf": "TF-IDF",
         "glove": "GloVe 100d", "word2vec_ag": "Word2Vec AG News", "word2vec_nyt": "Word2Vec NYT", "bert": "BERT"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manual_metrics(rows, classes):
    correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
    f1s = []
    for label in classes:
        tp = sum(r["true_label"] == label and r["predicted_label"] == label for r in rows)
        fp = sum(r["true_label"] != label and r["predicted_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["predicted_label"] != label for r in rows)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {"accuracy": correct/len(rows), "macro_f1": sum(f1s)/len(classes), "n": len(rows)}


def audit_results(output, source):
    meta = read_json(output / "split.json")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    if digest.hexdigest() != meta["data_sha256"]:
        raise ValueError("NYT source changed after the experiments.")
    results = {name: read_json(output / "metrics" / f"{name}.json") for name in NAMES}
    bert = results["bert"]
    if bert["config"]["epochs"] != 3 or bert["config"]["max_length"] != 64 or [r["epoch"] for r in bert["history"]] != [1, 2, 3]:
        raise ValueError("BERT results do not satisfy three epochs and max_length=64.")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        data = list(csv.DictReader(stream, strict=True))
    train_texts = {data[i]["text"].strip() for i in meta["indices"]["train"]}
    heldout = {i for i in meta["indices"]["test"] if data[i]["text"].strip() not in train_texts}
    sensitivity = {}
    for name, result in results.items():
        if result["split_hash"] != meta["split_hash"]:
            raise ValueError(f"Mismatched data split: {name}")
        with (output / "metrics" / f"{name}_predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(r["row_id"]) for r in rows] != meta["indices"]["test"]:
            raise ValueError(f"Prediction row IDs/order mismatch: {name}")
        if any(r["true_label"] != data[int(r["row_id"])]["label"] for r in rows):
            raise ValueError(f"Prediction ground truth mismatch: {name}")
        measured = manual_metrics(rows, meta["classes"])
        for key in ["accuracy", "macro_f1"]:
            if abs(measured[key]-result["test"][key]) > 1e-10:
                raise ValueError(f"Recomputed metrics disagree: {name} {key}")
        sensitivity[name] = manual_metrics([r for r in rows if int(r["row_id"]) in heldout], meta["classes"])
    (output / "duplicate_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    return meta, results, sensitivity


def chart(results):
    drawing = Drawing(480, 225)
    left, bottom, width, height = 145, 30, 310, 165
    for tick in [0, 25, 50, 75, 100]:
        x = left+width*tick/100
        drawing.add(Line(x, bottom, x, bottom+height, strokeColor=colors.HexColor("#dddddd")))
        drawing.add(String(x, bottom-15, str(tick), fontName="Helvetica", fontSize=8, textAnchor="middle"))
    for i, (name, result) in enumerate(results.items()):
        y = bottom+height-20-i*23
        drawing.add(String(left-8, y+1, NAMES[name], fontName="Helvetica", fontSize=9, textAnchor="end"))
        for delta, key, color in [(6, "accuracy", "#27577c"), (-3, "macro_f1", "#849caf")]:
            drawing.add(Rect(left, y+delta, width*result["test"][key], 7, fillColor=colors.HexColor(color), strokeColor=None))
    drawing.add(Rect(left, 211, 10, 8, fillColor=colors.HexColor("#27577c"), strokeColor=None))
    drawing.add(String(left+15, 211, "Accuracy", fontName="Helvetica", fontSize=9))
    drawing.add(Rect(left+95, 211, 10, 8, fillColor=colors.HexColor("#849caf"), strokeColor=None))
    drawing.add(String(left+110, 211, "Macro-F1", fontName="Helvetica", fontSize=9))
    drawing.add(String(455, 0, "%", fontName="Helvetica", fontSize=8, textAnchor="end"))
    return drawing


def build(output, source, destination):
    meta, results, sensitivity = audit_results(output, source)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("CN", str(font_path), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("CNBold", str(bold_path), subfontIndex=0))
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
        # Use an embedded Chinese TrueType font for best cross-platform rendering.
        global_font = "STSong-Light"
        pdfmetrics.registerFontFamily(global_font, normal=global_font, bold=global_font)
    regular = "CN" if font_path.exists() else "STSong-Light"
    bold = "CNBold" if font_path.exists() else regular
    pdfmetrics.registerFontFamily(regular, normal=regular, bold=bold, italic=regular, boldItalic=bold)
    styles = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=23, leading=34, textColor=colors.black, spaceAfter=12),
        "subtitle": ParagraphStyle("subtitle", fontName=regular, fontSize=11, leading=19, spaceAfter=18),
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=16, leading=24, spaceAfter=12, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=12, leading=19, spaceBefore=10, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("body", fontName=regular, fontSize=10, leading=17, spaceAfter=9, wordWrap="CJK"),
        "small": ParagraphStyle("small", fontName=regular, fontSize=8.5, leading=14, spaceAfter=7, wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName=regular, fontSize=9, leading=14, wordWrap="CJK"),
        "cellhead": ParagraphStyle("cellhead", fontName=bold, fontSize=9, leading=14, wordWrap="CJK"),
        "cellnum": ParagraphStyle("cellnum", fontName=regular, fontSize=9, leading=14, alignment=TA_CENTER),
    }
    story = []
    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))
    def heading(text):
        p(text, "h1")
    def table(rows, widths):
        cells = [[Paragraph(escape(str(c)), styles["cellhead" if i == 0 else "cellnum" if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", str(c)) else "cell"]) for c in row] for i, row in enumerate(rows)]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e7edf3")),
                              ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f7f9fb")]),
                              ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#d9d9d9")),
                              ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                              ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                              ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([t, Spacer(1,10)])
    def pct(value):
        return f"{100*value:.2f}%"
    def page():
        story.append(PageBreak())

    best = max(results, key=lambda n: results[n]["test"]["macro_f1"])
    test_counts = meta["class_counts"]["test"]
    majority = max(meta["class_counts"]["train"], key=meta["class_counts"]["train"].get)
    majority_acc = test_counts[majority]/sum(test_counts.values())
    baseline_f1 = (2*majority_acc/(1+majority_acc))/len(meta["classes"])
    p("作业一 文本分类实验报告", "title")
    p("词袋表示 词向量平均表示与 BERT 微调的对比", "subtitle")
    p(f"本实验在同一份 NYT 新闻数据划分上比较 7 种分类方法。全部方法均已实际训练并在同一个含 {meta['sizes']['test']:,} 条新闻的测试集上评价。"
      f"按测试集 Macro-F1 比较，{NAMES[best]} 最好，Accuracy 为 {pct(results[best]['test']['accuracy'])}，Macro-F1 为 {pct(results[best]['test']['macro_f1'])}。"
      "测试集仅用于最终评价，Logistic Regression 的 C 和 BERT 检查点均由验证集确定。")
    heading("1 数据集与实验划分")
    p("NYT 使用课程提供的 nyt.csv，字段为 text 与 label；AG News 使用课程提供的 ag.csv，仅包含 text，共 90,000 条。"
      "AG News 只用于训练外部 Word2Vec，分类任务的监督标签全部来自 NYT。")
    rows = [["类别", "全部", "训练集", "验证集", "测试集"]]
    for label in meta["classes"]:
        counts = [meta["class_counts"][s][label] for s in ["train", "validation", "test"]]
        rows.append([label, sum(counts), *counts])
    rows.append(["合计", sum(meta["sizes"].values()), *[meta["sizes"][s] for s in ["train", "validation", "test"]]])
    table(rows, [112,87,87,87,87])
    p("设置随机种子 42，使用 NumPy default_rng 生成一次随机排列，依次取 floor(0.8N) 为训练集、floor(0.1N) 为验证集，其余为测试集。"
      "保存原始行号和 split.json，后续所有方法复用该划分；不对类别做重采样。")
    p(f"体育类在全量数据中约占 75%。在测试集始终预测训练集多数类 {majority}，Accuracy 仍有 {pct(majority_acc)}，Macro-F1 只有 {pct(baseline_f1)}。"
      "因此，同时报告两个指标，并用验证集 Macro-F1 作为选模的主要依据。")
    p("评价定义", "h2")
    p("Accuracy = 正确预测数 / 样本总数。对每一类 k，F1(k) = 2TP(k) / [2TP(k) + FP(k) + FN(k)]；"
      "Macro-F1 是全部三个类别 F1 的算术平均。计算时显式传入完整类别列表，分母为零时记为 0。[1]", "small")
    page()

    heading("2 Task 1 词袋表示")
    p("全部传统方法先转为小写，再按英文单词及词内撇号分词；不去除停用词，不做词干化。"
      "词袋词表仅在 NYT 训练集构建，保留出现至少一次的单词，未设置词表上限。验证和测试仅执行 transform。")
    p("Binary Bag of Words", "h2")
    p("文档向量每个维度表示一个词是否出现。词出现一次或多次均记为 1，未出现记为 0。"
      "这一表示保留主题关键词的存在信息，弱化文档长度和同一词重复出现的影响。")
    p("Word Frequency", "h2")
    p("每个维度记录该词在当前文档中的出现次数。例如 apple apple banana 对应 apple 维度为 2、banana 维度为 1。"
      "计数表示保留了重复频率，但可能更受文档长度和高频词影响。")
    p("TF IDF", "h2")
    p("作业总述要求三种词袋方法，细则只列出 Binary BoW 和 Word Frequency，因此补充 TF-IDF。"
      "采用原始词频乘以平滑逆文档频率，idf(t) = log[(1 + N) / (1 + df(t))] + 1，再对文档向量作 L2 归一化。"
      "N 与 df(t) 都只由训练集估计。")
    p("Logistic Regression", "h2")
    p("三种词袋表示均使用带 L2 正则的多类 Logistic Regression，solver 为 lbfgs，max_iter 为 2000。"
      "候选 C 为 0.1、1、10，C 越大正则越弱。仅比较验证集 Macro-F1，若相同再比较 Accuracy；"
      "选出模型后不合并训练集和验证集重新拟合，直接评价测试集。")
    rows = [["表示", "词表大小", "选中 C", "验证 Macro-F1"]]
    for name in list(NAMES)[:3]:
        r = results[name]
        rows.append([NAMES[name], r["config"]["vocabulary_size"], r["config"]["C"], pct(r["validation"]["macro_f1"])])
    table(rows, [158,100,80,122])
    p("词袋模型保持稀疏矩阵形式，不将完整词表表示转成稠密矩阵，从而降低内存占用。"
      "每组模型连同向量器一起保存，可在新的文本上直接执行预测。")
    page()

    heading("3 Task 2 词向量平均表示")
    p("三组实验统一使用 100 维词向量。对文档中的每一次有效词出现取向量，求平均得到 100 维文档表示："
      "e(d) = [e(w1) + … + e(wn)] / n。n 只统计有向量的词，重复出现的词重复计入；全部词均未登录时使用零向量。")
    p("预训练 GloVe", "h2")
    p("使用 Stanford 官方 glove.6B 中的 glove.6B.100d.txt。该词向量固定，不在 NYT 上继续训练。"
      "程序只将当前语料会查询到的词向量载入内存，这只是从已有词典查表，不拟合验证集或测试集统计量。[2]")
    p("在 AG News 上训练 Word2Vec", "h2")
    p("使用全部 90,000 条 AG News 文本训练 Skip-gram Word2Vec，不读取任何 NYT 标签。"
      "训练后用词向量表示 NYT 文档，并在 NYT 训练集上训练分类器。")
    p("在 NYT 上训练 Word2Vec", "h2")
    p(f"仅使用 {meta['sizes']['train']:,} 条 NYT 训练文本训练 Word2Vec。验证集和测试集均不参与词向量训练。"
      "该组与 AG News 组使用相同超参数，以比较外部新闻语料和任务内语料的差异。")
    table([["参数", "两组 Word2Vec 共用取值"], ["维度与训练目标", "100 维 Skip-gram"], ["window 与 min_count", "5 与 2"],
           ["negative 与 epochs", "5 与 10"], ["workers 与 seed", "1 与 42"], ["实现", "gensim Word2Vec 固定词哈希"]], [160,300])
    p("gensim 训练使用单工作线程及固定初始化哈希，以减少多线程和 Python 哈希随机化带来的波动。[3]"
      "三组文档向量均先用训练集拟合 StandardScaler，再训练 Logistic Regression；验证与测试复用该缩放器，C 选择规则与 Task 1 相同。")
    rows = [["词向量来源", "训练词覆盖率", "测试词覆盖率", "测试零向量数", "选中 C"]]
    for name in ["glove", "word2vec_ag", "word2vec_nyt"]:
        r = results[name]
        rows.append([NAMES[name], pct(r["coverage"]["train"]["token_coverage"]), pct(r["coverage"]["test"]["token_coverage"]),
                     r["coverage"]["test"]["zero_vector_documents"], r["config"]["C"]])
    table(rows, [140,85,85,75,75])
    p("覆盖率按词出现次数计算，而非按唯一词数量计算。平均池化计算简便，但舍弃词序，并可能将少数判别性关键词的作用稀释。", "small")
    page()

    heading("4 Task 3 BERT 微调")
    bert = results["bert"]
    cfg = bert["config"]
    p("使用 google-bert/bert-base-uncased 及其 WordPiece 分词器，在预训练模型上添加三分类线性头，"
      "以交叉熵联合微调全部模型参数。BERT 接收原始文本，由其分词器处理大小写，不使用词袋分词后的输入。[4]")
    table([["设置", "本次取值"], ["预训练模型", "google-bert/bert-base-uncased"], ["最大序列长度", "64 包含 CLS 与 SEP"],
           ["训练轮数", "3 个完整 epoch"], ["优化器与学习率", "AdamW 与 2e-5"], ["weight_decay 与 warmup", "0.01 与总优化步数的 10%"],
           ["batch 与梯度累积", f"{cfg['batch_size']} 与 {cfg['gradient_accumulation']} 有效 batch 为 {cfg['batch_size']*cfg['gradient_accumulation']}"],
           ["设备与精度", f"{cfg.get('gpu') or cfg['device']} FP16"], ["检查点选择", f"验证 Macro-F1 最优 第 {cfg['selected_epoch']} 轮"]], [157,303])
    p("超过长度的文本截断，较短文本补齐到 64，并传入 attention_mask。每轮结束仅评价验证集；完成全部三轮后恢复验证集最优检查点，再评价一次测试集。"
      "使用梯度裁剪上限 1.0 和线性学习率衰减，梯度累积末尾不足一个完整窗口时按实际窗口大小缩放损失。")
    rows = [["Epoch", "训练平均 Loss", "验证 Accuracy", "验证 Macro-F1"]]
    for h in bert["history"]:
        rows.append([h["epoch"], f"{h['train_loss']:.4f}", pct(h["accuracy"]), pct(h["macro_f1"])])
    table(rows, [65,125,135,135])
    p(f"训练集与测试集中，原始 WordPiece 长度超过 64 的文本分别占 {pct(bert['truncation']['train']['fraction_over_64'])} 和 "
      f"{pct(bert['truncation']['test']['fraction_over_64'])}。这一长度限制遵循作业要求，但可能丢失新闻后部的主题线索；"
      "词袋与平均词向量方法使用全文，因此两类模型获得的文本范围并不相同。")
    page()

    heading("5 实验结果与比较")
    rows = [["方法", "测试 Accuracy", "测试 Macro-F1", "耗时 秒"]]
    for name, r in results.items():
        rows.append([NAMES[name], pct(r["test"]["accuracy"]), pct(r["test"]["macro_f1"]), f"{r['elapsed_seconds']:.1f}"])
    table(rows, [170,100,110,80])
    p("耗时包括本组表示构建、训练、验证选模及测试推断，不含资源下载；部分实验并发执行且 CPU 与 GPU 不同，耗时仅用于记录本次运行，不能据此作严格效率排名。", "small")
    story.extend([chart(results), Spacer(1, 10)])
    bow_best = max(list(NAMES)[:3], key=lambda n: results[n]["test"]["macro_f1"])
    emb_best = max(["glove", "word2vec_ag", "word2vec_nyt"], key=lambda n: results[n]["test"]["macro_f1"])
    p(f"词袋组中，{NAMES[bow_best]} 的测试 Macro-F1 为 {pct(results[bow_best]['test']['macro_f1'])}；"
      "三种词袋方法的结果接近，说明在这份三主题新闻数据中，词的出现信息已经具有很强的判别能力。"
      "该解释来自本次结果，未通过消融实验验证。")
    ag_score, nyt_score = results["word2vec_ag"]["test"]["macro_f1"], results["word2vec_nyt"]["test"]["macro_f1"]
    p(f"词向量组中，{NAMES[emb_best]} 最好。NYT Word2Vec 相对 AG News Word2Vec 的测试 Macro-F1 差值为 {(nyt_score-ag_score)*100:+.2f} 个百分点。"
      "语料规模、新闻领域和词覆盖率都可能影响这种差异；仅凭这两组结果，无法分别确定各因素的贡献。")
    p(f"BERT 的测试 Macro-F1 为 {pct(bert['test']['macro_f1'])}，与词袋组最优结果相差 "
      f"{(bert['test']['macro_f1']-results[bow_best]['test']['macro_f1'])*100:+.2f} 个百分点。"
      "预训练模型的复杂度并不保证在当前限制下得到最高分，64 token 截断、学习率与样本规模都值得在后续实验中单独检验。")
    page()

    heading("6 分类表现与重复文本检查")
    p(f"以下给出测试 Macro-F1 最优方法 {NAMES[best]} 的逐类别指标。类别支持数为该类在测试集中的真实样本数。")
    rows = [["类别", "Precision", "Recall", "F1", "支持数"]]
    for label in meta["classes"]:
        r = results[best]["classification_report"][label]
        rows.append([label, pct(r["precision"]), pct(r["recall"]), pct(r["f1-score"]), int(r["support"])])
    table(rows, [105,90,90,90,85])
    p("混淆矩阵 行是真实类别 列是预测类别", "h2")
    rows = [["真实 / 预测", *meta["classes"]]]
    for label, values in zip(meta["classes"], results[best]["confusion_matrix"]):
        rows.append([label, *values])
    table(rows, [130,110,110,110])
    overlap = meta["cross_split_duplicate_texts"]
    n_filtered = sensitivity[best]["n"]
    p(f"原始 NYT 存在 {meta['duplicate_texts']} 个重复文本记录。按去除首尾空白后的精确文本比较，训练与验证有 {overlap['train_validation']} 个共享文本，"
      f"训练与测试有 {overlap['train_test']} 个共享文本，验证与测试有 {overlap['validation_test']} 个共享文本。"
      "主实验保留原始记录并统一随机划分。")
    p(f"为检查重复文本影响，在原测试预测中排除与训练文本精确重复的记录，得到 {n_filtered} 条测试记录。"
      "本检查不重新训练、不重新选参数，故仅反映精确重复记录对已得到测试分数的影响。")
    rows = [["方法", "原测试 Macro-F1", "排除重复后 Macro-F1", "排除后 Accuracy"]]
    for name in NAMES:
        rows.append([NAMES[name], pct(results[name]["test"]["macro_f1"]), pct(sensitivity[name]["macro_f1"]), pct(sensitivity[name]["accuracy"])])
    table(rows, [155,95,115,95])
    p("本实验仅运行一个随机种子；小幅分差尚不能证明稳定优势。精确文本检查也不能发现近重复或转载改写。"
      "更严格的后续评估可在去重或按文章组划分后，用多个随机种子报告均值与标准差。", "small")
    page()

    heading("7 复现方式与结论")
    p("代码与输出", "h2")
    p("experiment.py 实现统一划分和全部七种方法；download_resources.py 获取指定预训练资源；build_report.py 从结果生成本报告。"
      "README.md 给出环境安装、分步骤运行和上传 GitHub 的说明，run_all.ps1 可在 Windows 一次执行完整流程。")
    p("outputs/split.json 保存划分索引、类别计数、输入 SHA-256 和运行环境；results.csv 汇总指标。"
      "outputs/metrics 保存每种方法的配置、验证候选值、分类报告、混淆矩阵及逐条测试预测。"
      "报告生成前重新读取真实标签与预测，独立重算两项指标并核对全部方法的测试行号及划分哈希。")
    p("复现命令", "h2")
    for text in ["python download_resources.py all", "python experiment.py --task all", "python build_report.py", "python -m pytest tests -q"]:
        p(escape(text), "small")
    p("实验环境", "h2")
    env = bert["environment"]
    p(f"Python {env['python']}；平台 {escape(env['platform'])}。", "small")
    table([["依赖", "本次版本"], *[[k, env["packages"][k]] for k in ["numpy", "scipy", "pandas", "scikit-learn", "gensim", "torch", "transformers"]]], [200,260])
    p("结论", "h2")
    p(f"在这份 NYT 新闻及固定 80/10/10 划分上，{NAMES[best]} 获得最高测试 Macro-F1。"
      "词袋模型利用主题关键词取得较高分类分数，平均词向量提供紧凑的文档表示，BERT 则通过上下文表示完成端到端微调。"
      "最终选择仍应考虑数据规模、输入长度、计算成本和误分类类型，本次结果不应直接推广到其他数据集。")
    page()

    heading("参考资料")
    refs = [
        ("[1] scikit-learn F1 score 文档", "https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html"),
        ("[2] Stanford GloVe 官方项目与预训练词向量", "https://nlp.stanford.edu/projects/glove/"),
        ("[3] gensim Word2Vec 参数与训练文档", "https://radimrehurek.com/gensim/models/word2vec.html"),
        ("[4] BERT-base-uncased 模型说明", "https://huggingface.co/google-bert/bert-base-uncased"),
    ]
    for title, url in refs:
        p(escape(title), "h2")
        p(f'<link href="{url}" color="#27577c">{escape(url)}</link>', "small")
    p("课程提供的 NYT 与 AG News CSV 是本实验的数据来源；未使用 AG News 的标签。"
      "本报告的数值来自 outputs/metrics 中保存的本次运行记录。")
    source_meta = cfg.get("pretrained_source", {})
    p("BERT 预训练权重版本", "h2")
    p(escape(source_meta.get("revision", "Hub default revision")), "small")
    p("资源完整性", "h2")
    p("GloVe 从 Stanford 官方压缩包按 HTTP Range 获取 100 维成员，解压后检查 ZIP 中记录的原始长度与 CRC。"
      "本次 GloVe 文本 SHA-256 为：", "small")
    p(results["glove"]["config"]["source_sha256"], "small")
    p("数据指纹", "h2")
    p(f"NYT SHA-256<br/>{meta['data_sha256']}<br/>划分 SHA-256<br/>{meta['split_hash']}", "small")
    p("测试集用于本报告的最终比较，不能再根据这些测试结果反复调整超参数。若改变参数、预处理或划分，应使用独立评估数据验证。", "small")

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=57, leftMargin=57,
                topMargin=48, bottomMargin=48, title="作业一 文本分类实验报告", author="", pageCompression=1)
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0]-57, 28, str(doc.page))
        canvas.restoreState()
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--nyt", type=Path, default=ROOT / "HW-1/nyt.csv")
    parser.add_argument("--pdf", type=Path, default=ROOT / "实验报告.pdf")
    args = parser.parse_args()
    build(args.output, args.nyt, args.pdf)
