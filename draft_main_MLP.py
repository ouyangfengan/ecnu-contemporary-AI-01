import re
import warnings
from email.parser import Parser
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'


def plot_best_model_loss(
    best_model, tuning_records,
    output_path=RESULTS_DIR / 'plots' / 'mlp_validation_loss.png',
    records_path=RESULTS_DIR / 'metrics' / 'mlp_best_loss_curve.csv',
):
    # 绘制真实训练损失轨迹，以及各参数配置的验证零一损失。
    pd.DataFrame({
        'epoch': range(1, len(best_model.loss_curve_) + 1),
        'training_loss_with_regularization': best_model.loss_curve_,
    }).to_csv(records_path, index=False)
    fig, (train_ax, val_ax) = plt.subplots(1, 2, figsize=(13, 5))
    train_ax.plot(
        range(1, len(best_model.loss_curve_) + 1), best_model.loss_curve_
    )
    train_ax.set_xlabel('Epoch')
    train_ax.set_ylabel('Training loss (including regularization)')
    train_ax.set_title('Best MLP: training loss')

    unique_records = [record for record in tuning_records if not record['reused']]
    labels = [
        f"{record['hidden_layer_sizes']}, alpha={record['alpha']}"
        for record in unique_records
    ]
    val_ax.bar(range(len(labels)), [
        1 - record['accuracy'] for record in unique_records
    ])
    val_ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    val_ax.set_ylabel('Validation zero-one loss (1 - accuracy)')
    val_ax.set_title('MLP: validation error by configuration')
    for ax in (train_ax, val_ax):
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"损失图已保存至：{output_path}；曲线数据：{records_path}", flush=True)
    return fig


def load_data():
    train_df = pd.read_csv(DATA_DIR / 'train_data.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled


def clean_text(text):
    """保留标题和正文，去除其他邮件头；无法识别结构时保留原文。"""
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'\n[ \t]*\n', normalized, maxsplit=1)
    if len(parts) != 2:
        return text

    header, body = parts
    lines = header.splitlines()

    if not lines or not re.match(
        r'^(From|Subject|Organization|Lines|Reply-To|NNTP-Posting-Host|'
        r'Newsgroups|Date|Message-ID|Path|Sender):', lines[0], re.IGNORECASE
    ):
        return text
    if not all(
        re.match(r'^[A-Za-z][A-Za-z0-9-]*:', line)
        or line.startswith((' ', '\t'))
        for line in lines
    ):
        return text

    headers = Parser().parsestr(header + '\n\n', headersonly=True)
    # 合并跨行标题，正文保持不变。
    subject = ' '.join(headers.get('Subject', '').split())
    return subject + '\n\n' + body if subject else body


X_train, y_train, X_test_unlabeled = load_data()

# 固定分层划分，便于与 LR 使用同一批训练和验证样本。
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"验证集样本数量: {len(X_val)}")
print(f"验证集标签数量: {len(y_val)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

X_train_clean = [clean_text(text) for text in X_train]
X_val_clean = [clean_text(text) for text in X_val]
X_test_clean = [clean_text(text) for text in X_test_unlabeled]

print("第一个训练样本清洗后（保留标题和正文）:")
print(X_train_clean[0])
print("-" * 20)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_clean)

# 先在 (100,) 上搜索 alpha，再固定最佳 alpha 比较四种隐藏层结构。
# 验证集仅使用训练集已拟合的 TF-IDF，不参与特征或模型拟合。
X_val_tfidf = vectorizer.transform(X_val_clean)
experiment_cache = {}
tuning_records = []
ALPHAS = [0.0001, 0.001, 0.01, 0.1]
BASELINE_LAYERS = (100,)
LAYERS_TO_COMPARE = [(50,), (100,), (200,), (100, 50)]


def evaluate_mlp(stage, alpha, hidden_layer_sizes=BASELINE_LAYERS):
    key = (alpha, hidden_layer_sizes)
    reused = key in experiment_cache
    if not reused:
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            alpha=alpha,
            max_iter=300,
            random_state=42,
        )
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X_train_tfidf, y_train)
        fit_seconds = perf_counter() - started
        converged = not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught
        )
        val_predictions = model.predict(X_val_tfidf)
        experiment_cache[key] = {
            'model': model,
            'alpha': alpha,
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': 'relu',
            'max_iter': 300,
            'actual_iterations': model.n_iter_,
            'converged': converged,
            'random_state': 42,
            'macro_f1': f1_score(
                y_val, val_predictions, average='macro', zero_division=0
            ),
            'accuracy': accuracy_score(y_val, val_predictions),
            'validation_log_loss': log_loss(
                y_val, model.predict_proba(X_val_tfidf), labels=model.classes_
            ),
            'fit_seconds': fit_seconds,
        }
    result = experiment_cache[key]
    record = {name: value for name, value in result.items() if name != 'model'}
    tuning_records.append({'stage': stage, **record, 'reused': reused})
    print(
        f"{stage} | alpha={alpha}, layers={hidden_layer_sizes}, "
        f"实际迭代={result['actual_iterations']}, 已收敛={result['converged']} | "
        f"Macro F1={result['macro_f1']:.6f}, "
        f"Accuracy={result['accuracy']:.6f}, "
        f"Validation log loss={result['validation_log_loss']:.6f}, "
        f"训练耗时={result['fit_seconds']:.2f}s"
        + ("（复用已有结果）" if reused else ""),
        flush=True,
    )
    if not result['converged']:
        print("提示：模型在 300 轮内尚未收敛，指标仅供参考。", flush=True)
    return result


def choose_best(results):
    # 按未四舍五入的 F1、准确率、训练耗时依次比较。
    return max(
        results,
        key=lambda result: (
            result['macro_f1'], result['accuracy'], -result['fit_seconds']
        ),
    )


print("\n--- 开始训练多层感知机 (MLP) 模型：搜索 alpha ---", flush=True)
baseline = evaluate_mlp('baseline', alpha=0.0001)
alpha_results = [
    evaluate_mlp('tune_alpha', alpha=alpha)
    for alpha in ALPHAS
]
best_alpha = choose_best(alpha_results)['alpha']
results = list(alpha_results)
print(f"\n--- 固定 alpha={best_alpha}，对照隐藏层结构 {LAYERS_TO_COMPARE} ---", flush=True)
results.extend(
    evaluate_mlp('tune_layers', alpha=best_alpha, hidden_layer_sizes=layers)
    for layers in LAYERS_TO_COMPARE
)
best_result = choose_best(results)
print("模型训练完成！", flush=True)

print("\n--- 最终对照 ---")
for name, result in [('默认 MLP 基线', baseline), ('最优方案', best_result)]:
    print(
        f"{name}: alpha={result['alpha']}, "
        f"layers={result['hidden_layer_sizes']}, "
        f"实际迭代={result['actual_iterations']}, "
        f"Macro F1={result['macro_f1']:.6f}, "
        f"Accuracy={result['accuracy']:.6f}, "
        f"Validation log loss={result['validation_log_loss']:.6f}"
    )

best_params = {
    'alpha': best_result['alpha'],
    'hidden_layer_sizes': best_result['hidden_layer_sizes'],
    'activation': 'relu',
    'max_iter': best_result['max_iter'],
    'random_state': 42,
}
print(f"最终选择：{best_params}")
print(f"相对默认 MLP 基线的 Macro F1 变化："
      f"{best_result['macro_f1'] - baseline['macro_f1']:+.6f}")
pd.DataFrame(tuning_records).to_csv(
    RESULTS_DIR / 'metrics' / 'mlp_tuning_results.csv', index=False
)

# 复用训练部分上拟合的最佳模型；本步骤尚不进行全量数据重训。
mlp_classifier = best_result['model']

X_test_tfidf = vectorizer.transform(X_test_clean)
y_pred_mlp = mlp_classifier.predict(X_test_tfidf)

pd.DataFrame(y_pred_mlp).to_csv(
    RESULTS_DIR / 'predictions' / 'mlp_predictions.csv',
    index=False, header=False,
)

# 先保存预测和图片，再显示窗口，避免窗口阻塞文件输出。
loss_figure = plot_best_model_loss(mlp_classifier, tuning_records)
if plt.get_backend().lower() != 'agg':
    plt.show()
plt.close(loss_figure)
