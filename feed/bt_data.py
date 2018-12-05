from backtrader.feeds import GenericCSVData


#              0       1       2       3        4      5         6               7           8           9
BASIC_COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]


class BTAbstractCSVData(GenericCSVData):
    @staticmethod
    def get_params(columns):
        params = [('nullvalue', float('NaN')), ('dtformat', '%Y-%m-%d'), ('tmformat', '%H:%M:%S'), ('headers', False)]

        for i in range(len(columns)):
            p = (columns[i].lower(), i)
            params.append(p)

        params.append(('time', -1))
        params.append(('openinterest', -1))

        return tuple(params)


class BTCSVBasicData(BTAbstractCSVData):
    lines = ('price_change', 'p_change', 'turnover',)

    params = BTAbstractCSVData.get_params(BASIC_COLUMNS)
    # params = (
    #     ('nullvalue', float('NaN')),
    #     ('dtformat', '%Y-%m-%d'),
    #     ('tmformat', '%H:%M:%S'),
    #
    #     ('datetime', 0),
    #     ('time', -1),
    #     ('open', 1),
    #     ('high', 2),
    #     ('close', 3),
    #     ('low', 4),
    #     ('volume', 5),
    #     ('openinterest', -1),
    #     ('price_change', 6),
    #     ('p_change', 7),
    #     ('turnover', 8),
    # )



