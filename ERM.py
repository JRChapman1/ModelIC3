# Import external libraries
import pandas as pd
import numpy as np
from scipy import stats

# Import custom classes
from Mortality import Mortality
from InterestRates import InterestRate


class ERM:
    def __init__(self, int_rate_curve, steps_py=1, mortality_stress=None, lapse_rate=None):

        self.steps_py = steps_py

        # Create instance of Mortality class
        self.mdl_mortality = Mortality(steps_py, mortality_stress, lapse_rate)

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
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy().transpose()

        # Calculate the rolled up loan values at each future time-step for each LTM and store in a numpy array
        np_loan_vals = self.rolled_up_loan_values(policy_data)

        # Calculate the projected property values at each future time-step for each LTM and store in a numpy array
        np_prop_vals = self.projected_property_values(policy_data, hpi_drift, prop_shock)

        # For each LTM, at each future time-step, calculate the intrinsic NNEG as the shortfall in the projected
        # property value below the rolled up loan amount
        np_loan_vals = np.maximum(np_loan_vals - np_prop_vals, 0)

        # Multiply the projected intrinsic NNEG at each time-step for each LTM by the probability of the LTM being
        # redeemed during that time-step and store the result in a dataframe
        df_redemption_vals = pd.DataFrame(np_loan_vals * np_death_in_period_probs)

        return df_redemption_vals

    def expected_mkt_nneg_val(self, policy_data, hpi_drift, hpi_vol, prop_shock, aggregate=False):

        # Apply shock to property values if required
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)

        # Store the projected market NNEG values for each LTM in a DataFrame
        df_mkt_nneg = self.projected_nneg_mkt_value(policy_data, hpi_drift, hpi_vol, prop_shock).to_numpy()

        # For each LTM, store the probability of redemption in each year in a DataFrame
        df_death_in_period_probs = self.mdl_mortality.prob_death()
        df_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].transpose().to_numpy()

        # Multiply the projected market NNEG values by the redeption probabilities to get the expected market NNEG
        # cashflow for each policy at each time-step
        df_mkt_nneg = pd.DataFrame(df_mkt_nneg * df_death_in_period_probs)

        # If required, sum the expected market NNEG cashflows for each policy at each time-step
        if aggregate:
            df_mkt_nneg = df_mkt_nneg.sum(axis='columns')

        return df_mkt_nneg

    def rolled_up_loan_values(self, policy_data):

        # Store the AERs of each LTM in a (1 x n) numpy array
        np_roll_up_rates = policy_data['AER'].to_numpy().reshape((1, -1))

        # Construct a numpy array whose (i,j)th entry is the accumulation factor for the loan on LTM j at time-step i
        np_roll_up_facs = (1 + np_roll_up_rates) ** np.array([t / self.steps_py for t in range(0, 105 * self.steps_py + 1)]).reshape(-1, 1)

        # Multiply the accumulation factors by the initial loan values and return the result
        return np_roll_up_facs * policy_data['Loan Value'].to_numpy()

    def projected_property_values(self, policy_data, hpi_drift, prop_shock):

        # Apply shock to property values if required
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)

        # Construct a numpy array whose (i,j)th entry is the growth factor for the property backing LTM j at time-step i
        np_growth_facs = (1 + hpi_drift) ** np.array([t / self.steps_py for t in range(0, 105 * self.steps_py + 1)]).reshape(-1, 1)

        # Multiply the growth factors by the initial property values and return the result
        return np_growth_facs * policy_data['Property Value'].to_numpy()

    def expected_redemption_cfs(self, policy_data):

        # Store the rolled up loan values for each LTM at each time-step in a numpy array
        np_loan_vals = self.rolled_up_loan_values(policy_data)

        # Store the probability of each LTM being redeemed at each time-step in a numpy array
        df_death_in_period_probs = self.mdl_mortality.prob_death()
        np_death_in_period_probs = df_death_in_period_probs.loc[policy_data['Age']].to_numpy().transpose()

        # For each LTM, multiply the loan value at each time-step by the probability of the LTM being redeemed at that
        # time and return the result
        return pd.DataFrame(np_loan_vals * np_death_in_period_probs)

    def expected_redemption_cfs_net_of_mkt_nneg(self, policy_data, hpi_drift, hpi_vol, prop_shock):

        # Return the expected redemption cashflows minus the expected market NNEG cashflows
        return self.expected_redemption_cfs(policy_data) - self.expected_mkt_nneg_val(policy_data, hpi_drift, hpi_vol, prop_shock)

    def projected_nneg_mkt_value(self, policy_data, hpi_drift, hpi_vol, prop_shock):

        # Apply shock to property values if required
        if prop_shock is not None:
            policy_data['Property Value'] *= (1 - prop_shock)

        # Store the rolled up loan values for each LTM at each time-step in a numpy array
        np_loan_vals = self.rolled_up_loan_values(policy_data)

        # Store the projected property values for each LTM at each time-step in a numpy array
        np_prop_vals = self.projected_property_values(policy_data, hpi_drift, prop_shock)

        # Construct a numpy array whose entries are the time-steps under consideration
        np_time_steps = np.array(range(0, 105 * self.steps_py + 1)).reshape(-1, 1) / self.steps_py

        # For each LTM, and at each possible time-step, use Black-76 to calculate the market value of the NNEG assuming
        # that the LTM is redeemed at that time
        np_log_prop_loan_ratio = np.log(np_prop_vals / np_loan_vals)
        np_d1 = (np_log_prop_loan_ratio + 0.5 * hpi_vol**2 * np_time_steps) / (hpi_vol * np.sqrt(np_time_steps))
        np_d2 = np_d1 - hpi_vol * np.sqrt(np_time_steps)
        np_nneg_mkt_val = np_loan_vals * stats.norm.cdf(-np_d2) - np_prop_vals * stats.norm.cdf(-np_d1)

        return pd.DataFrame(np_nneg_mkt_val)


    def value(self, policy_data, hpi_drift, hpi_vol, prop_shock, rebase_step=0, aggregate=True):

        # Project the expected redemption cashflows (net of market NNEG) for each LTM
        exp_red_cfs = self.expected_redemption_cfs_net_of_mkt_nneg(policy_data, hpi_drift, hpi_vol, prop_shock)

        # For each LTM, construct a discount curve from the risk-free rate of interest plus the day-one IFRS spread
        # on the LTM
        disc_curve = self.mdl_interest.discount_curve(rebase_step, policy_data['Spread']).iloc[:exp_red_cfs.shape[0]]

        # Calculate the present values of the expected redemption cashflows for each LTM using the corresponding
        # discount curve and sum the
        df_ltm_values = (exp_red_cfs * disc_curve).sum(axis='columns')

        # If required, aggregate NNEG values over all time-steps
        if aggregate:
            df_ltm_values = df_ltm_values.sum(axis='rows')

        return df_ltm_values

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
    df_ltm_pols = pd.DataFrame([[74, 1, 1 / 0.4, 0.0603, 0], [75, 1, 1 / 0.4, 0.0603, 0], [76, 1, 1 / 0.4, 0.0603, 0]], columns=['Age', 'Loan Value', 'Property Value', 'AER', 'Spread'])
    EqRelModel = ERM('flat_spot_4pc', 1)
    foo = EqRelModel.expected_redemption_cfs_net_of_mkt_nneg(df_ltm_pols, 0.03, 0.12, None)
    s = EqRelModel.spread(df_ltm_pols, foo)
    bar = EqRelModel.expected_redemption_cfs_net_of_int_nneg(df_ltm_pols, 0.03, None)
    print(s)

