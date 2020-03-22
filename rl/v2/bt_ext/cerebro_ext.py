import backtrader as bt


class RLCerebro(bt.Cerebro):
    BASIC_COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER",
                     "LABEL"]

    def __init__(self):
        super(RLCerebro, self).__init__()
        self._agent = None

    def addagent(self, agent):
        self._agent = agent

    def getagent(self):
        return self._agent

    def run(self):
        if self._agent is None:
            raise ValueError

        bt.Cerebro.run(self)
