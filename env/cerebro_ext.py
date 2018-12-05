import backtrader as bt


class RLExtCerebro(bt.Cerebro):
    def __init__(self):
        bt.Cerebro.__init__(self)

        self._agent = None

    def addagent(self, agent):
        self._agent = agent

    def getagent(self):
        return self._agent

    def run(self):
        if self._agent is None:
            raise NotImplementedError

        bt.Cerebro.run(self)
