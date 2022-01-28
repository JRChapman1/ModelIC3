# Import external libraries
import pandas as pd
import numpy as np


class Mortality:

    def __init__(self, steps_py, mortality_stress=None, lapse_rate=None):

        self.steps_py = steps_py
        self.lapse_rate = lapse_rate

        # Read in mortality data and set age as index
        self.mort_data = pd.read_csv('static/inputs/game_mort_data.csv')
        self.mort_data.set_index('Age', inplace=True)

        # Apply mortality stress if applicable
        self.mort_data = apply_stress(self.mort_data, mortality_stress)

    def q(self, x):

        # Construct empty dataframe whose index is the list of ages of interest (in the chosen time-step)
        x = pd.DataFrame(x, columns=['Age'])

        # Calculate the age of interest for each life in years from their age in the chosen time-step
        x['Age Year'] = (x['Age']/self.steps_py).astype(int)

        # Read the annual qx value of each age of interest from the base annual qx curve
        x = x.merge(self.mort_data, left_on='Age Year', right_index=True, how='left')

        # Convert the annual qx values into values expressed in terms of the required step size and return the result
        return (1 + x['Death Prob']) ** (1/self.steps_py) - 1

    def prob_in_force(self):

        # Get the qx values for all possible ages of interest
        qx_curve = self.q([x for x in range(17*self.steps_py, 121*self.steps_py)])

        # Calculate the px curve for all possible ages of interest from the qx curve
        px_curve = (1 - qx_curve).to_list()

        # Construct a DataFrame whose (i, j)th entry is p(i+j)
        # TODO: Refactor
        df_px_arr = pd.DataFrame([[1] + px_curve[i:] + [0] * i for i in range(0, len(px_curve))], index=[x for x in range(17*self.steps_py, 121*self.steps_py)])

        # Adjust px values for lapses
        # TODO: Refactor
        if self.lapse_rate is not None:
            for i in range(0, 104):
                df_px_arr[i] -= self.lapse_rate[i]

        # Calculate the cumulative product of the px values in each row of the DataFrame constructed above and return the result
        return df_px_arr.cumprod(axis=1)

    def prob_death(self):

        # Get the probabilities of lives of each possible age being inforce at each point in time
        df_prob_in_force = self.prob_in_force()

        # Get the qx values for all possible ages of interest
        qx_curve = (self.q([x for x in range(17*self.steps_py, 121*self.steps_py)])).to_list()

        # Construct a DataFrame whose (i, j)th entry is q(i+j)
        # TODO: Refactor
        df_qx = pd.DataFrame([qx_curve[i:] + [0] * (i+1) for i in range(0, len(qx_curve))], index=[x for x in range(17*self.steps_py, 121*self.steps_py)])

        # Adjust qx values for lapses
        # TODO: Refactor
        if self.lapse_rate is not None:
            for i in range(0, 104):
                df_qx[i] += self.lapse_rate[i]

        # Multiply the probability inforce at each time by the probability of dying during that time-step
        df_prob_death =  df_qx * df_prob_in_force

        # Increment columns by 1,  set probs to 0 at time 0, and return the result
        df_prob_death.columns += 1
        df_prob_death[0] = 0.0
        return df_prob_death[[col for col in range(0, df_prob_death.columns.max() + 1)]]

    def simulate_mortality(self, policy_data):

        # Simulate a random number between 0 and 1 for each in-force policy
        np_test_stats = np.random.uniform(size=policy_data.shape[0])

        # Get the qx value for each policy
        qx = self.q(policy_data['Age'])

        # Flag each policy as still in-force if and only if the simulated random value is greater than the qx
        # probability for that policy
        np_pol_if_flags = (np_test_stats > qx).to_numpy()

        # Return only the policies from the inputted policy data whose in-force flag is true
        return policy_data[np_pol_if_flags]


def apply_stress(base_mortality_curve, stress):
    if stress is not None:

        # Read in the multiplicative stress factors for each stress
        mort_stress_factors = pd.read_csv('static/inputs/mortality_stress_factors.csv', index_col='Stress')

        # Multiply the original qx probabilities by the multiplicative stress factor for the chosen stress
        base_mortality_curve['Death Prob'] *= mort_stress_factors.loc[stress, 'Factor']

    # Subject all qx probabilities to a cap of 1 and a floor of 0 and return the result
    return base_mortality_curve.clip(0, 1)


if __name__ == '__main__':
    model = Mortality(1, 'SII')
    foo = model.prob_death()
    print(foo)
    foo = model.prob_in_force()
    print(foo)