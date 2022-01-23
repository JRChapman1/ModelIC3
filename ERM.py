# Import external libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# Import custom classes
from Mortality import Mortality
from InterestRates import InterestRate


class ERM:
    def __init__(self, int_rate_curve, steps_py=1, mortality_stress=None):

        self.steps_py = steps_py

        # Create instance of Mortality class
        self.mdl_mortality = Mortality(steps_py, mortality_stress)

        # Create instance of InterestRate class
        self.mdl_interest = InterestRate(int_rate_curve, steps_py, None)

    def expected_redemption_cfs_net_of_int_nneg(self, policy_data, hpi_drift, prop_shock):

        # Subtract expected intrinsic NNEG cashflows from expected redemption cashflows and return the result
        return self.expected_redemption_cfs(policy_data) - self.expected_int_nneg_cfs(policy_data, hpi_drift, prop_shock)


    def expected_int_nneg_cfs(self, policy_data, hpi_drift, prop_shock):

        # Apply shock to property values if required
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)

        # For each LTM, get probabilities of policy being redeemed in each future time-step and store in a numpy array
        df_death_in_period_probs = self.mdl_mortality.prob_death()
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy()

        # Calculate the rolled up loan values at each future time-step for each LTM and store in a numpy array
        np_roll_up_rates = policy_data['AER'].to_numpy().reshape((-1, 1))
        np_roll_up_facs = (1 + np_roll_up_rates) ** np.array(df_death_in_period_probs.columns / self.steps_py).reshape(1, -1)
        np_loan_vals = np_roll_up_facs * policy_data['Loan Value'].to_numpy().reshape(-1, 1)

        # Calculate the projected property values at each future time-step for each LTM and store in a numpy array
        np_prop_vals = policy_data['Property Value'].to_numpy().reshape((-1, 1)) * np.repeat(1 + hpi_drift, 105).cumprod()

        # For each LTM, at each future time-step, calculate the intrinsic NNEG as the shortfall in the projected
        # property value below the rolled up loan amount
        np_loan_vals = np.maximum(np_loan_vals-np_prop_vals, 0)

        # Multiply the projected intrinsic NNEG at each time-step for each LTM by the probability of the LTM being
        # redeemed during that time-step and store the result in a dataframe
        df_redemption_vals = pd.DataFrame(np_loan_vals * np_death_in_period_probs)

        # Increment the dataframe columns by 1
        # TODO: Why is this necessary?
        df_redemption_vals.columns += 1

        # Transpose the dataframe and return the result
        # TODO: Why transpose necessary?
        return df_redemption_vals.transpose()

    def expected_mkt_nneg_cfs(self, policy_data, hpi_drift, hpi_vol, prop_shock, aggregate=False):

        # Apply shock to property values if required
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)

        df_mkt_nneg = self.projected_nneg_mkt_value(policy_data, hpi_drift, hpi_vol, prop_shock).to_numpy()
        df_death_in_period_probs = self.mdl_mortality.prob_death()
        df_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].transpose().to_numpy()
        df_mkt_nneg = pd.DataFrame(df_mkt_nneg * df_death_in_period_probs)
        if aggregate:
            df_mkt_nneg = df_mkt_nneg.sum(axis='columns')
        return df_mkt_nneg

    def rolled_up_loan_values(self, policy_data):
        np_roll_up_rates = policy_data['AER'].to_numpy().reshape((-1, 1))
        np_roll_up_facs = (1 + np_roll_up_rates) ** np.array([t / self.steps_py for t in range(0, 105 * self.steps_py)]).reshape(1, -1)
        return np_roll_up_facs * policy_data['Loan Value'].to_numpy().reshape(-1, 1)

    def projected_property_values(self, policy_data, hpi_drift, prop_shock):

        # Apply shock to property values if required
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)
        # Calculate the projected property values at each future time-step for each LTM and store in a numpy array
        return policy_data['Property Value'].to_numpy().reshape((-1, 1)) * np.repeat(1 + hpi_drift, 105).cumprod()

    def expected_redemption_cfs(self, policy_data):
        np_loan_vals = self.rolled_up_loan_values(policy_data)
        df_death_in_period_probs = self.mdl_mortality.prob_death()
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy()
        df_redemption_vals = pd.DataFrame(np_loan_vals * np_death_in_period_probs)
        df_redemption_vals.columns += 1
        return df_redemption_vals.transpose()

    def expected_redemption_cfs_net_of_mkt_nneg(self, policy_data, hpi_drift, hpi_vol, prop_shock):
        return self.expected_redemption_cfs(policy_data) - self.expected_mkt_nneg_cfs(policy_data, hpi_drift, hpi_vol, prop_shock)

    def projected_nneg_mkt_value(self, policy_data, hpi_drift, hpi_vol, prop_shock):
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)
        np_loan_vals = self.rolled_up_loan_values(policy_data).transpose()
        np_prop_vals = self.projected_property_values(policy_data, hpi_drift, prop_shock).transpose()
        np_log_prop_loan_ratio = np.log(np_prop_vals / np_loan_vals)
        np_time_steps = np.array(range(0, 105 * self.steps_py)).reshape(-1, 1) / self.steps_py
        np_d1 = (np_log_prop_loan_ratio + (hpi_drift + 0.5 * hpi_vol**2) * np_time_steps) / (hpi_vol * np.sqrt(np_time_steps))
        np_d2 = np_d1 - hpi_vol * np.sqrt(np_time_steps)
        np_nneg_mkt_val = np_loan_vals * stats.norm.cdf(-np_d2) - np_prop_vals * np.exp(-hpi_drift * np_time_steps) * stats.norm.cdf(-np_d1)
        return pd.DataFrame(np_nneg_mkt_val)


    def value(self, policy_data, hpi_drift, hpi_vol, prop_shock, rebase_step=0, aggregate=True):
        exp_red_cfs = self.expected_redemption_cfs_net_of_mkt_nneg(policy_data, hpi_drift, hpi_vol, prop_shock)
        disc_curve = self.mdl_interest.discount_curve(rebase_step, policy_data['Spread']).iloc[:exp_red_cfs.shape[0]]
        if aggregate:
            return (exp_red_cfs * disc_curve).sum(axis='columns').sum(axis='rows')
        else:
            return (exp_red_cfs * disc_curve).sum(axis='columns')

    def spread(self, policy_data, cfs, itterations=10):
        spread = policy_data['Spread']
        disc_curve = self.mdl_interest.discount_curve(spread=spread).iloc[:cfs.shape[0]]
        disc_curve.index = cfs.index
        duration = (cfs * disc_curve.multiply(disc_curve.index, axis=0)).sum() / (cfs * disc_curve).sum() / self.steps_py
        for i in range(1, itterations):
            disc_curve = self.mdl_interest.discount_curve(spread=spread).iloc[:cfs.shape[0]]
            spread -= (policy_data['Loan Value'] / (cfs * disc_curve).sum() - 1) / duration
            disc_curve = self.mdl_interest.discount_curve(spread=spread).iloc[:cfs.shape[0]]
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
    df_ltm_pols = pd.DataFrame([[75, 1, 1 / 0.4, 0.07, 0]], columns=['Age', 'Loan Value', 'Property Value', 'AER', 'Spread'])
    EqRelModel = ERM('flat_spot_4pc', 1)
    foo = EqRelModel.projected_nneg_mkt_value(df_ltm_pols, 0.03, 0.12, None)
    bar = EqRelModel.expected_int_nneg_cfs(df_ltm_pols, 0.03, None)

    print(foo)

