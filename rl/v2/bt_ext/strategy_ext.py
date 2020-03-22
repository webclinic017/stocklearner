import backtrader as bt


class RLCommonStrategy(bt.Strategy):
    params = (
        ("printlog", True),
        ("time_step", 0)
    )

    def log(self, txt, dt=None, doprint=False):
        """ Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the schema[0] dataseries
        self.dataclose = self.datas[0].close
        self.p_change = self.datas[0].p_change

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.date = None

        self.current_value = 0.0
        self.reward = 0.0
        self.done = False
        self.last_observation = None
        self.current_observation = None
        self.next_observation = None

        self.agent = self.env.getagent()
        self.action = self.agent.Hold

    def start(self):
        self.log("start -> info")

    def prenext_open(self):
        self.log("prenext_open -> info")

    def next_open(self):
        self.log("next_open -> info")

    def nextstart_open(self):
        self.log("nextstart_open -> info")

    def close(self, data=None, size=None, **kwargs):
        self.log("close -> info")

    def notify_order(self, order):
        # self.log("notify_order " + str(order.status))
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

    def _get_reward(self):
        self.log("_get_reward")
        return 0.0

    def _get_observation(self, date):
        self.log(date)
        return None, False

    # As bt_ext.render(), but need to get observation and reward
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("next -> Close, %.2f" % self.dataclose[0])
        self.log("next -> Broker value: " + str(round(self.broker.getvalue(), 2)) + ", cash: " + str(
            round(self.broker.get_cash(), 2)))
        self.log("next -> Position size: " + str(self.position.size) + ", price: " + str(round(self.position.price, 2)))

        self.date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        # print("date in next()")
        # print(self.date)

        # Fetch observation, do the rest thing first which should be done in step function
        self.last_observation = self.current_observation
        self.current_value = round(self.broker.getvalue(), 2)
        self.current_observation, self.done = self._get_observation(date=self.date)
        self.reward = self._get_reward()
        self.action = self.agent.choose(self.current_observation)

        if self.action == self.agent.Hold:
            pass
        elif self.action == self.agent.Buy:
            # BUY, BUY, BUY!!! (with all possible default parameters)
            self.log("BUY CREATE, %.2f" % self.dataclose[0], doprint=False)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()

        elif self.action == self.agent.Sell and self.position:
            # We must in the market before we can sell
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.log("SELL CREATE, %.2f" % self.dataclose[0], doprint=False)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell()
