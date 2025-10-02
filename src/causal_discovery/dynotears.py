import numpy as np
import re

import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic


def dynotears(process, max_lags):

    df = pd.DataFrame(process.T, columns=[f'X{i}' for i in range(len(process))])
    sm = from_pandas_dynamic(df, p=max_lags)

    cm = np.zeros((process.shape[0], process.shape[0]))

    for cause in sm:
        sources = sm[cause]
        for source in sources:
            weight = sources[source]['weight']
            x_source, lag_source = source.split('_')
            x_cause, lag_cause = cause.split('_')
            x_source = int(re.sub("[^0-9]", "", x_source))
            x_cause = int(re.sub("[^0-9]", "", x_cause))
            lag_source = int(re.sub("[^0-9]", "", lag_source))
            lag_cause = int(re.sub("[^0-9]", "", lag_cause))
            lag = abs(lag_cause - lag_source)
            if lag > 0 and x_source != x_cause:
                cm[x_source, x_cause] = max(cm[x_source, x_cause], weight)
        
    return np.where(cm != 0, 1, 0).astype(int)