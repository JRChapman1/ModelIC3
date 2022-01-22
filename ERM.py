import pandas as pd
import numpy as np
from Mortality import Mortality
from InterestRates import InterestRate
from BusinessOperations import BusinessOperations as BizOps
from matplotlib import pyplot as plt
from LapseRisk import LapseRisk
from scipy import stats


class ERM:
    def __init__(self, int_rate_curve, steps_py, hpi_assumption=0.03, prop_shock=None, mortality_stress=None, lapse_rate=None):
        self.steps_py = steps_py
        self.BaseMortality = Mortality(mortality_stress, steps_py, lapse_rate)
        self.BaseIR = InterestRate(int_rate_curve, steps_py, None)
        self.hpi_assumption = hpi_assumption
        self.prop_shock = prop_shock

    def expected_redemption_cfs_net_of_int_nneg(self, policy_data):
        if self.prop_shock is not None:
            policy_data['Property Value'] *= (1 - self.prop_shock)
        df_death_in_period_probs = self.BaseMortality.prob_death()
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy()
        np_roll_up_rates = policy_data['AER'].to_numpy().reshape((-1, 1))
        np_roll_up_facs = (1 + np_roll_up_rates) ** np.array(df_death_in_period_probs.columns / self.steps_py).reshape(1, -1)
        np_loan_vals = np_roll_up_facs * policy_data['Loan Value'].to_numpy().reshape(-1, 1)
        np_prop_vals = policy_data['Property Value'].to_numpy().reshape((-1, 1)) * np.repeat(1 + self.hpi_assumption, 105).cumprod()
        np_loan_vals = np.minimum(np_loan_vals, np_prop_vals)
        df_redemption_vals = pd.DataFrame(np_loan_vals * np_death_in_period_probs)
        df_redemption_vals.columns += 1
        return df_redemption_vals.transpose()

    def expected_int_nneg_cfs(self, policy_data):
        if self.prop_shock is not None:
            policy_data['Property Value'] *= (1 - self.prop_shock)
        df_death_in_period_probs = self.BaseMortality.prob_death()
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy()
        np_roll_up_rates = policy_data['AER'].to_numpy().reshape((-1, 1))
        np_roll_up_facs = (1 + np_roll_up_rates) ** np.array(df_death_in_period_probs.columns / self.steps_py).reshape(1, -1)
        np_loan_vals = np_roll_up_facs * policy_data['Loan Value'].to_numpy().reshape(-1, 1)
        np_prop_vals = policy_data['Property Value'].to_numpy().reshape((-1, 1)) * np.repeat(1 + self.hpi_assumption, 105).cumprod()
        np_loan_vals = np.maximum(np_loan_vals-np_prop_vals, 0)
        df_redemption_vals = pd.DataFrame(np_loan_vals * np_death_in_period_probs)
        df_redemption_vals.columns += 1
        return df_redemption_vals.transpose()


    def rolled_up_loan_values(self, policy_data):
        if self.prop_shock is not None:
            policy_data['Property Value'] *= (1 - self.prop_shock)
        np_roll_up_rates = policy_data['AER'].to_numpy().reshape((-1, 1))
        np_roll_up_facs = (1 + np_roll_up_rates) ** np.array([t / self.steps_py for t in range(0, 105 * self.steps_py)]).reshape(1, -1)
        return np_roll_up_facs * policy_data['Loan Value'].to_numpy().reshape(-1, 1)

    def expected_redemption_cfs(self, policy_data):
        np_loan_vals = self.rolled_up_loan_values(policy_data)
        df_death_in_period_probs = self.BaseMortality.prob_death()
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy()
        df_redemption_vals = pd.DataFrame(np_loan_vals * np_death_in_period_probs)
        df_redemption_vals.columns += 1
        return df_redemption_vals.transpose()

    def projected_nneg_mkt_value(self, policy_data, hpi_drift=0.03, hpi_vol=0.12):
        np_loan_vals = self.rolled_up_loan_values(policy_data).transpose()
        np_prop_vals = policy_data['Property Value'].to_numpy()
        np_log_prop_loan_ratio = np.log(np_prop_vals / np_loan_vals)
        np_time_steps = np.array(range(0, 105 * self.steps_py)).reshape(-1, 1) / self.steps_py
        np_d1 = (np_log_prop_loan_ratio + (hpi_drift + 0.5 * hpi_vol**2) * np_time_steps) / (hpi_vol * np.sqrt(np_time_steps))
        np_d2 = np_d1 - hpi_vol * np.sqrt(np_time_steps)
        np_nneg_mkt_val = np_loan_vals * stats.norm.cdf(-np_d2) - np_prop_vals * np.exp(-hpi_drift * np_time_steps) * stats.norm.cdf(-np_d1)
        return pd.DataFrame(np_nneg_mkt_val)

    def spread(self, pv, cfs, itterations=10):
        spread = 0
        disc_curve = self.BaseIR.discount_curve(spread=spread).iloc[:cfs.shape[0]]
        duration = (cfs[0] * disc_curve['Rate'] * disc_curve.index).sum() / (cfs[0] * disc_curve['Rate']).sum() / self.steps_py
        for i in range(1, itterations):
            disc_curve = self.BaseIR.discount_curve(spread=spread).iloc[:cfs.shape[0]]
            spread -= (pv / (cfs[0] * disc_curve['Rate']).sum() - 1) / duration
            disc_curve = self.BaseIR.discount_curve(spread=spread).iloc[:cfs.shape[0]]
            duration = (cfs[0] * disc_curve['Rate'] * disc_curve.index).sum() / (cfs[0] * disc_curve['Rate']).sum() / self.steps_py
        return spread

    def value(self, policy_data, rebase_step):
        exp_red_cfs = self.expected_redemption_cfs(policy_data)
        disc_curve = self.BaseIR.discount_curve(rebase_step, float(policy_data['Spread'])).iloc[:exp_red_cfs.shape[0]]
        return (exp_red_cfs[0] * disc_curve['Rate']).sum()

    def value2(self, policy_data, rebase_step, aggregate=True):
        exp_red_cfs = self.expected_redemption_cfs(policy_data)
        disc_curve = self.BaseIR.discount_curve2(rebase_step, policy_data['Spread']).iloc[:exp_red_cfs.shape[0]]
        if aggregate:
            return (exp_red_cfs * disc_curve).sum().sum()
        else:
            return (exp_red_cfs * disc_curve).sum()

    def spread2(self, policy_data, cfs, itterations=10):
        spread = policy_data['Spread']
        disc_curve = self.BaseIR.discount_curve2(spread=spread).iloc[:cfs.shape[0]]
        disc_curve.index = cfs.index
        duration = (cfs * disc_curve.multiply(disc_curve.index, axis=0)).sum() / (cfs * disc_curve).sum() / self.steps_py
        for i in range(1, itterations):
            disc_curve = self.BaseIR.discount_curve2(spread=spread).iloc[:cfs.shape[0]]
            spread -= (policy_data['Loan Value'] / (cfs * disc_curve).sum() - 1) / duration
            disc_curve = self.BaseIR.discount_curve2(spread=spread).iloc[:cfs.shape[0]]
            duration = (cfs * disc_curve.multiply(disc_curve.index, axis=0)).sum() / (cfs * disc_curve).sum() / self.steps_py
        return spread


def plot_results(data, title=None):
    for column in data:
        plt.plot(data.index, data[column], label=column)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == '__main__':
    df_ltm_pols = pd.DataFrame([[70, 1, 1 / 0.4, 0.07, 0], [65, 1, 1 / 0.7, 0.04, 0], [70, 1, 1 / 0.4, 0.07, 0]], columns=['Age', 'Loan Value', 'Property Value', 'AER', 'Spread'])
    EqRelModel = ERM('flat_spot_4pc', 1)
    nneg = EqRelModel.projected_nneg_mkt_value(df_ltm_pols)
    foo = EqRelModel.expected_redemption_cfs_net_of_int_nneg(df_ltm_pols)
    print('done')

