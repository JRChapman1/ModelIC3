import pandas as pd

from Assets import Portfolio
from Liabilities import Liabilities
from Annuity import Annuity
from ERM import ERM
from BusinessOperations import BusinessOperations as BizOps
from matplotlib import pyplot as plt
from InterestRates import InterestRate

import warnings

class InsuranceCompany:
    def __init__(self, cash, int_rate_curve, hpi_assumption, ann_portfolio, ann_margin, erm_portfolio, write_ann_nb, write_erm_nb, incl_deferreds, prop_reins_perc, steps_py):

        self.cash = cash
        self.int_rate_curve = int_rate_curve
        self.hpi_assumption = hpi_assumption
        self.steps_py = steps_py
        self.ann_portfolio = ann_portfolio
        self.ann_margin = ann_margin
        self.erm_portfolio = erm_portfolio
        self.write_ann_nb = write_ann_nb
        self.write_erm_nb = write_erm_nb
        self.incl_deferreds = incl_deferreds
        self.prop_reins_perc = prop_reins_perc
        self.AnnuityModel = Annuity(int_rate_curve, steps_py)
        self.EqRelModel = ERM(int_rate_curve, steps_py)
        self.IntRateModel = InterestRate(int_rate_curve, steps_py, None)


    def run_single_projection(self, projection_steps):
        LiabilityModel = Liabilities(self.ann_portfolio.copy(), self.ann_margin, self.int_rate_curve, self.prop_reins_perc, self.steps_py)
        if self.erm_portfolio is not None:
            erm_outgo = self.erm_portfolio['Loan Value'].sum()
        else:
            erm_outgo = 0
        self.cash += LiabilityModel.premiums() - erm_outgo
        AssetModel = Portfolio(self.cash, self.erm_portfolio, self.int_rate_curve, self.hpi_assumption, self.steps_py)
        df_balance_sheet = pd.DataFrame(columns=['Capital', 'BE Liabs', 'SII Liabs'], index=range(0, projection_steps + 1))
        for i in range(0, 41):
            df_balance_sheet.loc[i] = [AssetModel.value(i), LiabilityModel.best_est_value(i), LiabilityModel.sii_value(i)]
            AssetModel.increment(i)
            LiabilityModel.increment()
            ann_outgo = LiabilityModel.outgo_for_period()
            AssetModel.cash -= ann_outgo
            if self.write_ann_nb:
                first_pol_index = LiabilityModel.ann_portfolio.index.max() + 1
                ann_nb = BizOps.write_annuity_business(self.write_ann_nb, self.incl_deferreds, first_pol_index, self.steps_py)
                AssetModel.cash += self.AnnuityModel.value(ann_nb, i+1, 'Base') * (1 + self.ann_margin)
                LiabilityModel.ann_portfolio = LiabilityModel.ann_portfolio.append(ann_nb)
            if self.write_erm_nb:
                first_pol_index = AssetModel.erm_portfolio.index.max() + 1
                erm_nb = BizOps.write_ltm_business(self.write_erm_nb, first_pol_index, self.steps_py)
                exp_red_cfs = self.EqRelModel.expected_redemption_cfs(erm_nb)
                spreads = self.EqRelModel.spread2(erm_nb, exp_red_cfs)
                erm_nb['Spread'] = spreads.transpose()
                #AssetModel.cash -= self.EqRelModel.value2(erm_nb, i)
                AssetModel.erm_portfolio = AssetModel.erm_portfolio.append(erm_nb)

        return df_balance_sheet

    def project_cashflows(self, projection_steps, aggregate=False):
        res = pd.DataFrame()
        if self.erm_portfolio is not None:
            res['ERMs'] = self.EqRelModel.expected_redemption_cfs(self.erm_portfolio).sum(axis=1)
        if self.ann_portfolio is not None:
            res['Annuities'] = self.AnnuityModel.expected_cfs(self.ann_portfolio).sum(axis=1)

        return res

    def write_and_value_nb(self, num_pols, incl_deferreds, rebase_step, steps_py):
        first_pol_index = self.ann_portfolio.index.max() + 1
        nb = BizOps.write_annuity_business(num_pols, incl_deferreds, first_pol_index, self.steps_py)
        nb_value = self.AnnuityModel.value(nb, rebase_step, 'Base') * (1 + self.ann_margin)
        self.ann_portfolio = self.ann_portfolio.append(nb)
        return nb_value

def plot_results(data, title=None):
    for column in data:
        plt.plot(data.index, data[column], label=column)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()

def run_multiple_projections(projection_steps, num_projections):
    warnings.filterwarnings("ignore")
    ann_pols = BizOps.write_annuity_business(10000, steps_py=1)
    erm_pols = BizOps.write_ltm_business(1000, steps_py=1)
    IC = InsuranceCompany(0, 'boe_spot_annual', 0.03, ann_pols, 0, erm_pols, 0, 0, 0, 0, 1)
    single_proj = IC.run_single_projection(projection_steps)
    df_avg_bs = single_proj / num_projections
    for proj in range(2, num_projections + 1):
        print(f'Running projection {proj}')
        IC = InsuranceCompany(0, 'boe_spot_annual', 0.03, ann_pols, 0, None, 0, 0, 0, 0, 1)
        df_avg_bs += IC.run_single_projection(projection_steps) / num_projections
    return df_avg_bs

if __name__ == '__main__':
    ann_pols = BizOps.write_annuity_business(1000, steps_py=1)
    erm_pols = BizOps.write_ltm_business(1000, steps_py=1)
    IC = InsuranceCompany(0, 'boe_spot_annual', 0.03, ann_pols, 0, erm_pols, 0, 0, 0, 0, 1)
    res = IC.run_single_projection(30)
    plot_results(res)
