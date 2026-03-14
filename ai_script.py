# ============================================
# CDT501 作业示例：使用 Transformers 做文本情感分析
# 并用 Matplotlib 可视化情感置信度柱状图
# 代码适合直接在 Google Colab 中运行
# ============================================

# -------- 第一步：安装依赖（在 Colab 中执行一次即可）---------
# 如果你在本地环境运行，可以把下面两行复制到终端执行：
# pip install transformers torch matplotlib

# 在 Colab 里用 "!" 调用 shell 命令
!pip install -q transformers torch matplotlib

# -------- 第二步：导入所需库 --------
# 从 transformers 库中导入 pipeline，用于快速创建情感分析流程
from transformers import pipeline  # 创建情感分析管道 [web:1][web:7][web:10]

# 导入 matplotlib 的 pyplot 子库，用于绘图
import matplotlib.pyplot as plt    # 绘制柱状图 [web:5]

# -------- 第三步：创建情感分析模型 --------
# 使用 Hugging Face 的预训练模型，指定任务为 "sentiment-analysis"
# 不指定模型名称时，会默认下载一个在英文情感数据集（如 SST-2）上微调好的模型 [web:10]
sentiment_analyzer = pipeline("sentiment-analysis")

# -------- 第四步：准备需要分析的文本列表 --------
# 你可以根据作业或自己的需要，把下面的句子改成中文或英文的任意文本
# 注意：transformers 默认使用英文情感模型，中文效果可能一般，但可以用于演示
texts = [
    "I really love this course, it is very helpful!",
    "This assignment is too difficult and I feel frustrated.",
    "The weather is okay, not good, not bad.",
    "The movie was fantastic, I enjoyed every minute.",
    "The service at the restaurant was terrible."
]

# -------- 第五步：对所有文本进行情感分析 --------
# pipeline 支持直接对一个字符串列表进行批量分析 [web:1][web:7]
results = sentiment_analyzer(texts)

# results 是一个列表，每个元素是形如：
# {'label': 'POSITIVE', 'score': 0.998...}
# 的字典，label 表示情感标签，score 表示置信度

# 为了后续处理和可视化，我们把标签和置信度分别提取出来
labels = []   # 存放情感标签（Positive / Negative）
scores = []   # 存放置信度分数

for res in results:
    # 原始标签通常是 'POSITIVE' 或 'NEGATIVE'
    raw_label = res["label"]

    # 为了输出更符合题目要求，统一转成首字母大写的形式
    # 如果标签是 POSITIVE -> Positive, NEGATIVE -> Negative
    if raw_label.upper() == "POSITIVE":
        label = "Positive"
    elif raw_label.upper() == "NEGATIVE":
        label = "Negative"
    else:
        # 少见情况：有的模型可能给出 NEUTRAL 等其他标签，这里原样输出
        label = raw_label.capitalize()

    labels.append(label)
    scores.append(res["score"])

# -------- 第六步：在终端（控制台）打印情感分析结果 --------
print("=== 文本情感分析结果 ===")
for text, label, score in zip(texts, labels, scores):
    # 使用 :.4f 的格式控制，只保留 4 位小数，输出更美观
    print(f"文本: {text}")
    print(f"情感标签: {label}, 置信度: {score:.4f}")
    print("-" * 50)

# -------- 第七步：使用 Matplotlib 绘制情感置信度柱状图 --------
# 思路：
# x 轴：每一段文本，可以用简短编号（Text 1, Text 2, ...）
# y 轴：对应文本的情感置信度 scores
# 同时用颜色区分 Positive / Negative，便于视觉理解 [web:5]

# 生成 x 轴标签，例如 ["Text 1", "Text 2", ...]
x_labels = [f"Text {i+1}" for i in range(len(texts))]

# 为不同情感类型准备颜色映射，Positive 用蓝色，Negative 用红色
colors = []
for label in labels:
    if label == "Positive":
        colors.append("tab:blue")
    elif label == "Negative":
        colors.append("tab:red")
    else:
        # 其他情感（例如 Neutral）用灰色
        colors.append("gray")

# 创建一个新的图像窗口
plt.figure(figsize=(10, 6))  # 设置图像大小，宽 10、高 6（单位：英寸）

# 绘制柱状图，x 轴为文本编号，y 轴为置信度，color 为颜色列表
plt.bar(x_labels, scores, color=colors)

# 设置图表标题和坐标轴标签
plt.title("Sentiment Analysis Confidence Scores")
plt.xlabel("Text Index")
plt.ylabel("Confidence Score")

# 在柱子顶部标注具体数值（保留两位小数），便于阅读
for x, y in zip(x_labels, scores):
    plt.text(
        x,                 # x 位置：柱子的 x 坐标
        y + 0.01,          # y 位置：稍微高于柱顶一点点
        f"{y:.2f}",        # 显示两位小数
        ha="center",       # 水平居中
        va="bottom",       # 垂直底部对齐
        fontsize=10
    )

# 让 x 轴标签倾斜一点，避免文字重叠（文本太多时很有用）
plt.xticks(rotation=15)

# 设置 y 轴范围在 [0, 1]，因为置信度是 0~1 之间的概率
plt.ylim(0, 1.05)

# 自动调整子图边距，防止标签被遮挡
plt.tight_layout()

# 显示图像
plt.show()
