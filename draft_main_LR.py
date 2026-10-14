import re
import warnings
from email.parser import Parser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from time import perf_counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'


def plot_best_model_loss(
    best_model, train_features, train_labels, val_features, val_labels,
    output_path=RESULTS_DIR / 'plots' / 'lr_best_loss_curve.png',
    records_path=RESULTS_DIR / 'metrics' / 'lr_best_loss_curve.csv',
):
    # 固定最优 C，以独立训练的迭代预算绘制对数损失曲线。
    final_iterations = int(np.max(best_model.n_iter_))
    budgets = [
        budget for budget in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
        if budget < final_iterations
    ]
    budgets.append(max(1, final_iterations))
    records = []
    for index, budget in enumerate(budgets):
        if index == len(budgets) - 1:
            model = best_model
            converged = True
        else:
            # 非终点预算独立重训，不是一次拟合的逐轮损失。
            model = clone(best_model).set_params(max_iter=budget)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(train_features, train_labels)
            converged = not any(
                issubclass(warning.category, ConvergenceWarning)
                for warning in caught
            )
        train_loss = log_loss(
            train_labels, model.predict_proba(train_features), labels=model.classes_
        )
        val_loss = log_loss(
            val_labels, model.predict_proba(val_features), labels=model.classes_
        )
        records.append({
            'iteration_budget': budget,
            'actual_max_iterations': int(np.max(model.n_iter_)),
            'train_log_loss': train_loss,
            'validation_log_loss': val_loss,
            'converged': converged,
            'reused_best_model': model is best_model,
            'C': model.C,
        })
        print(
            f"迭代预算={budget} | Train log loss={train_loss:.6f} | "
            f"Validation log loss={val_loss:.6f} | 已收敛={converged}",
            flush=True,
        )

    results = pd.DataFrame(records)
    results.to_csv(records_path, index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        results['iteration_budget'], results['train_log_loss'],
        marker='o', label='Training (measured)',
    )
    ax.plot(
        results['iteration_budget'], results['validation_log_loss'],
        marker='s', label='Validation (measured)',
    )
    ax.set_xscale('log')
    ax.set_xlabel('Iteration budget (log scale; independent fits)')
    ax.set_ylabel('Log loss (without regularization)')
    ax.set_title(f'Best Logistic Regression: measured loss by iteration budget\nC={best_model.C}')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(
        0.5, 0.02,
        'Fixed hyperparameters; not an epoch trace. Last point reuses the best model.',
        ha='center', fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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

# 固定分层划分，便于与 MLP 使用同一批训练和验证样本。
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

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_clean)

# 固定特征提取方案，仅搜索逻辑回归的正则化参数 C。
# 验证集仅使用训练集已拟合的 TF-IDF，不参与特征或模型拟合。
X_val_tfidf = vectorizer.transform(X_val_clean)
experiment_cache = {}
tuning_records = []


def fit_until_converged(C):
    """只有求解器报告未收敛时，才增加迭代上限并重新拟合。"""
    max_iter = 1000
    while True:
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X_train_tfidf, y_train)
        fit_seconds = perf_counter() - started
        if not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught
        ):
            return model, max_iter, fit_seconds
        if max_iter >= 8000:
            raise RuntimeError(
                f"C={C} 在 max_iter={max_iter} 时仍未收敛，请检查数据或求解器。"
            )
        print(f"C={C} 在 {max_iter} 次内未收敛，增加迭代上限后重试。", flush=True)
        max_iter *= 2


def evaluate_lr(stage, C):
    reused = C in experiment_cache
    if not reused:
        model, max_iter, fit_seconds = fit_until_converged(C)
        val_predictions = model.predict(X_val_tfidf)
        experiment_cache[C] = {
            'model': model,
            'C': C,
            'max_iter': max_iter,
            'actual_max_iterations': int(np.max(model.n_iter_)),
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
    result = experiment_cache[C]
    record = {name: value for name, value in result.items() if name != 'model'}
    tuning_records.append({'stage': stage, **record, 'reused': reused})
    print(
        f"{stage} | C={C}, max_iter={result['max_iter']}, "
        f"实际迭代={result['actual_max_iterations']} | "
        f"Macro F1={result['macro_f1']:.6f}, "
        f"Accuracy={result['accuracy']:.6f}, "
        f"Validation log loss={result['validation_log_loss']:.6f}, "
        f"训练耗时={result['fit_seconds']:.2f}s"
        + ("（复用已有结果）" if reused else ""),
        flush=True,
    )
    return result


def choose_best(results):
    # 按未四舍五入的 F1、准确率、训练耗时依次比较。
    return max(
        results,
        key=lambda result: (
            result['macro_f1'], result['accuracy'], -result['fit_seconds']
        ),
    )


print("\n--- 开始训练逻辑回归模型：搜索 C ---", flush=True)
baseline = evaluate_lr('baseline', C=1)
results = [
    evaluate_lr('tune_C', C=C)
    for C in [0.1, 1, 10, 100]
]
best_result = choose_best(results)
print("模型训练完成！", flush=True)

print("\n--- 最终对照 ---")
for name, result in [('C=1 基线', baseline), ('最优方案', best_result)]:
    print(
        f"{name}: C={result['C']}, max_iter={result['max_iter']}, "
        f"Macro F1={result['macro_f1']:.6f}, "
        f"Accuracy={result['accuracy']:.6f}, "
        f"Validation log loss={result['validation_log_loss']:.6f}"
    )

best_params = {
    'C': best_result['C'],
    'max_iter': best_result['max_iter'],
    'random_state': 42,
}
print(f"最终选择：{best_params}")
print(f"相对 C=1 基线的 Macro F1 变化："
      f"{best_result['macro_f1'] - baseline['macro_f1']:+.6f}")
pd.DataFrame(tuning_records).to_csv(
    RESULTS_DIR / 'metrics' / 'lr_tuning_results.csv', index=False
)

# 复用训练部分上拟合的最佳模型；本步骤尚不进行全量数据重训。
lr_classifier = best_result['model']

X_test_tfidf = vectorizer.transform(X_test_clean)
y_pred_lr = lr_classifier.predict(X_test_tfidf)

pd.DataFrame(y_pred_lr).to_csv(
    RESULTS_DIR / 'predictions' / 'lr_predictions.csv',
    index=False, header=False,
)

# 先保存预测和图片，再显示窗口，避免窗口阻塞文件输出。
loss_figure = plot_best_model_loss(
    lr_classifier, X_train_tfidf, y_train, X_val_tfidf, y_val,
)
plt.show()
plt.close(loss_figure)
