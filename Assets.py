from ERM import ERM
from InterestRates import InterestRate
from Mortality import Mortality
from BusinessOperations import BusinessOperations as BizOps
from matplotlib import pyplot as plt


class Portfolio:
    def __init__(self, cash, erm_portfolio, int_rate_curve, hpi_assumption, steps_py):
        self.cash = cash
        self.erm_portfolio = erm_portfolio
        self.hpi_assumption = hpi_assumption
        self.steps_py = steps_py

        self.EqRelModel = ERM(int_rate_curve, steps_py)
        self.BaseIR = InterestRate(int_rate_curve, steps_py, False)
        self.BaseMortality = Mortality(False, steps_py)

        self.forward_curve = self.BaseIR.forward_curve()
        if erm_portfolio is not None:
            exp_erm_red_cfs = self.EqRelModel.expected_redemption_cfs(self.erm_portfolio)
            self.erm_portfolio['Spread'] = self.EqRelModel.spread2(self.erm_portfolio, exp_erm_red_cfs).transpose()

    def value(self, rebase_step):
        if self.erm_portfolio is not None:
            erm_val = self.EqRelModel.value2(self.erm_portfolio, rebase_step)
        else:
            erm_val = 0
        return self.cash + erm_val

    def increment(self, rebase_step):
        if self.erm_portfolio is not None:
            next_erm_portfolio = self.BaseMortality.simulate_mortality(self.erm_portfolio)
            redemptions = self.erm_portfolio[~self.erm_portfolio.index.isin(next_erm_portfolio.index)][['Property Value', 'Loan Value']].min(axis=1).sum()
            self.erm_portfolio = next_erm_portfolio
            self.erm_portfolio['Age'] += 1
            self.erm_portfolio['Loan Value'] *= (1 + self.erm_portfolio['AER']) ** (1 / self.steps_py)
            self.erm_portfolio['Property Value'] *= (1 + self.hpi_assumption) ** (1 / self.steps_py)
            # TODO: Loan value should not be changed because NNEG is biting - should be dealt with in ERM value function
            self.erm_portfolio['Loan Value'] = self.erm_portfolio[['Loan Value', 'Property Value']].values.min(axis=1)
        else:
            redemptions = 0
        self.cash = (self.cash + redemptions) * (1 + float(self.forward_curve.iloc[rebase_step]))


if __name__ == '__main__':
    steps_py = 1
    policy = BizOps.write_ltm_business(10000, steps_py=1)
    res = []
    p = Portfolio(1000000000, None, 'boe_spot_annual', 0.03, steps_py)
    for i in range(0, 51):
        res.append(p.value(i))
        p.increment(i)
    plt.plot(range(0, 51), res)
    plt.show()