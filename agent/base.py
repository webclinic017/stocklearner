import numpy as np
import backtrader as bt


class RLBaseAgent:
    def __init__(self):
        self.Hold, self.Buy, self.Sell = range(3)
        self.actions = [self.Hold, self.Buy, self.Sell]

    def sample(self):
        idx = np.random.random_integers(0, len(self.actions) - 1)
        return self.actions[idx]

    def choose_action(self, observation):
        raise NotImplementedError


class RLAgent01(RLBaseAgent):
    def __init__(self):
        RLBaseAgent.__init__(self)

    def choose_action(self, observation):
        # TODO
        return self.sample()


class RLCommonStrategy(bt.Strategy):
    params = (
        ("printlog", False),
    )

    def log(self, txt, dt=None, doprint=False):
        """ Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # self.p_change = self.datas[0].p_change

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.agent = self.env.getagent()
        self.observation = None

    # As env.step(), but next observation, reward and done will be given in next function.
    def notify_order(self, order):
        self.log("notify_order " + str(order.status))
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm),
                    doprint=False)

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log("SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm),
                         doprint=False)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None

    # As env.render(), but need to get observation and reward
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("Next Close, %.2f" % self.dataclose[0])

        # Fetch observation
        self.observation = None

        action = self.agent.choose_action(self.observation)

        if action == self.agent.Hold:
            pass
        elif action == self.agent.Buy:
            # BUY, BUY, BUY!!! (with all possible default parameters)
            self.log("BUY CREATE, %.2f" % self.dataclose[0], True)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()

        elif action == self.agent.Sell:
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.log("SELL CREATE, %.2f" % self.dataclose[0], True)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell()
