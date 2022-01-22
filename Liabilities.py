from Annuity import Annuity
from InterestRates import InterestRate
from Mortality import Mortality
from BusinessOperations import BusinessOperations as BizOps
from matplotlib import pyplot as plt


class Liabilities:
    def __init__(self, ann_portfolio, ann_margin, int_rate_curve, prop_reins_perc, steps_py):
        self.ann_portfolio = ann_portfolio
        self.ann_margin = ann_margin
        self.prop_reins_perc = prop_reins_perc
        self.steps_py = steps_py

        self.AnnuityModel = Annuity(int_rate_curve, steps_py)
        self.InterestRateModel = InterestRate(int_rate_curve, steps_py, False)
        self.MortalityModel = Mortality(False, steps_py)

        self.ann_portfolio['Annual Amt'] *= (1-prop_reins_perc)

    def premiums(self):
        ann_prems = self.best_est_value(0) * (1 + self.ann_margin)
        return ann_prems

    def best_est_value(self, rebase_step):
        ann_val = self.AnnuityModel.value(self.ann_portfolio, rebase_step, stress='Base')
        return ann_val

    def sii_value(self, rebase_step):
        ann_val = self.AnnuityModel.solvency_value(self.ann_portfolio, rebase_step)
        return ann_val

    def outgo_for_period(self):
        ann_outgo = self.ann_portfolio[self.ann_portfolio['Age'] > 65][ 'Annual Amt'].sum() / self.steps_py
        return ann_outgo

    def increment(self):
        self.ann_portfolio = self.MortalityModel.simulate_mortality(self.ann_portfolio)
        self.ann_portfolio['Age'] += 1


if __name__ == '__main__':
    ann_pols = BizOps.write_annuity_business(1000, True, steps_py=1)
    L = Liabilities(ann_pols, 0, 'eiopa_spot_annual', 0.5,  1)
    best_est = []
    sii = []
    out = L.outgo_for_period()
    for i in range(0, 41):
        print(i)
        best_est.append(L.best_est_value(i))
        sii.append(L.sii_value(i))
        L.increment()
    plt.plot(range(0, 41), best_est)
    plt.plot(range(0, 41), sii)
    plt.show()