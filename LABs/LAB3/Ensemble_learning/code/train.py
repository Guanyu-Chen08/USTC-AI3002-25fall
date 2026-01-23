import os
import joblib
from datetime import datetime

from arguments import get_args
from dataloader import load_data
from evaluator import evaluate
from submission import VotingModel, BaggingModel, AdaBoostModel, GBDTModel, StackingModel
from visualization import plot_confusion, plot_adaboost_training_curve
from sklearn.ensemble import GradientBoostingClassifier

def run_one(method, X_train, X_test, y_train, y_test, args):
    print(f"\n===== Running {method} =====")

    # 选择模型
    if method == "voting":
        model = VotingModel()
    elif method == "bagging":
        model = BaggingModel(n_estimators=args.n_estimators, max_depth=args.max_depth)
    elif method == "adaboost":
        model = AdaBoostModel(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate
        )
    elif method == "gbdt":
        # model = GBDTModel(
        #     n_estimators=args.n_estimators,
        #     learning_rate=args.learning_rate
        # )
        model = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=args.n_estimators, subsample=1.0
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=1
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )
    elif method == "stacking":
        model = StackingModel()
    else:
        raise ValueError(f"Unknown method: {method}")

    # 训练
    model.fit(X_train, y_train)

    # 评估
    pred = model.predict(X_test)
    acc, f1 = evaluate(model, X_test, y_test)
    print(f"{method} → Acc={acc:.4f}, F1={f1:.4f}")

    # 创建保存目录
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, stamp + f"_{method}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    joblib.dump(model, os.path.join(save_dir, f"{method}.pkl"))

    # 保存混淆矩阵
    plot_confusion(
        y_test, pred,
        title=f"{method} Confusion Matrix",
        save_path=os.path.join(save_dir, f"{method}_cm.png")
    )

    # AdaBoost 特殊：保存训练误差曲线
    if method == "adaboost" and hasattr(model, "train_errors"):
        plot_adaboost_training_curve(
            model.train_errors,
            save_path=os.path.join(save_dir, "adaboost_training_curve.png")
        )

    return acc, f1

def main():
    args = get_args()
    X_train, X_test, y_train, y_test = load_data()

    # 不指定方法时，默认跑完全部
    if args.method == "all":
        methods = ["voting", "bagging", "adaboost", "gbdt", "stacking"]
        acc_results = {}
        f1_results = {}
        for method in methods:
            acc, f1 = run_one(method, X_train, X_test, y_train, y_test, args)
            acc_results[method] = acc
            f1_results[method] = f1
        print("\n===== Generating Comparison Plots =====")

        from visualization import plot_model_comparison
        
        plot_model_comparison(
            acc_results, 
            save_path=os.path.join(args.save_dir, "comparison_accuracy.png"),
            method='accuracy'
        )
        
        plot_model_comparison(
            f1_results, 
            save_path=os.path.join(args.save_dir, "comparison_f1.png"),
            method='f1'
        )
        print("Comparison plots saved!")
        
    else:
        run_one(args.method, X_train, X_test, y_train, y_test, args)


if __name__ == "__main__":
    main()
