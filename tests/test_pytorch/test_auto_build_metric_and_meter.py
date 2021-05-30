import flame


def test_build_metric():
    cfg = {
        '_type': 'flame.pytorch.metrics.CopyMetric',
        'name': 'loss'
    }
    metric = flame.auto_builder.build_from_config(cfg)
    assert metric

    output = {
        'loss': 1
    }
    metrics = metric(output)
    assert metrics['loss'] == output['loss']


def test_build_metric_list():
    cfg = {
        '_type': 'flame.pytorch.metrics.MetricList',
        'metrics': [
            {
                '_type': 'flame.pytorch.metrics.CopyMetric',
                'name': 'loss',
            },
            {
                '_type': 'flame.pytorch.metrics.Topk'
            }
        ]
    }