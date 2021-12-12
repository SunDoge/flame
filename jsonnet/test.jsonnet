local flame = import './flame.libsonnet';

assert flame.normLr(0.1, 256) == 0.1 : 'wrong lr';

true
