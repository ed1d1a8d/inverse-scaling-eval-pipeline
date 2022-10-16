rm -rv results/testing

python -m eval_pipeline.main \
    --dataset-path "data/test/classification.csv" \
    --exp-dir testing \
    --models opt-125m opt-350m babbage curie \
    --task-type classification \
    --batch-size 100

python -m eval_pipeline.plot_loss \
    --no-show --task-type classification_acc testing

python -m eval_pipeline.plot_loss \
    --no-show --task-type classification_loss testing

python -m eval_pipeline.plot_loss \
    --no-show --task-type classification_partial testing
