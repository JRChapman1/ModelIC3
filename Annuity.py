import pandas as pd
import numpy as np
from Mortality import Mortality
from InterestRates import InterestRate
from math import sqrt
from matplotlib import pyplot as plt


class Annuity:
    def __init__(self, int_rate_curve, steps_py):
        self.steps_py = steps_py
        self.BaseMortality = Mortality(steps_py, None)
        self.StressMortality = Mortality(steps_py, 'SII')
        self.BaseIR = InterestRate(int_rate_curve, steps_py, None)
        self.StressIR = InterestRate(int_rate_curve, steps_py, 'SII')
        self.annuity_factors_cache = {}

    def annuity_factors(self, rebase_step, stress):

        if f'{stress}_{rebase_step}' not in self.annuity_factors_cache:
            # Set in-force probabilities depending on whether mortality is being stressed or not
            if stress == 'Mortality':
                df_if_probs = self.StressMortality.prob_in_force()
            else:
                df_if_probs = self.BaseMortality.prob_in_force()

            # Set discount curve depending on whether interest rates are being stressed or not
            if stress == 'Interest':
                discount_curve = self.StressIR.discount_curve(rebase_step)
            else:
                discount_curve = self.BaseIR.discount_curve(rebase_step)

            # Increment columns of in-force probs dataframe so that they align with discount curve index
            df_if_probs.columns += 1
            df_payable = np.reshape(df_if_probs.index, (-1, 1)) + np.reshape(df_if_probs.columns, (1, -1)) > 65 * self.steps_py
            df_if_probs.mask(~df_payable, 0, inplace=True)
            df_if_probs = df_if_probs.dot(discount_curve[:df_if_probs.shape[1]]) / self.steps_py
            df_if_probs.columns = ['Ann Fac']
            self.annuity_factors_cache[f'{stress}_{rebase_step}'] = df_if_probs

        return self.annuity_factors_cache[f'{stress}_{rebase_step}']

    def value(self, policy_data, rebase_step, stress):
        df_ann_facs = self.annuity_factors(rebase_step, stress)
        df_liab_calc = policy_data.merge(df_ann_facs, left_on='Age', right_index=True, how='left')
        df_liab_calc['Policy Value'] = df_liab_calc['Ann Fac'] * df_liab_calc['Annual Amt']
        return df_liab_calc['Policy Value'].sum()

    def expected_cfs(self, policy_data):
        df_if_probs = self.BaseMortality.prob_in_force().loc[policy_data['Age']]
        df_if_probs.index = policy_data.index
        return df_if_probs.transpose() * policy_data['Annual Amt']


    def solvency_value(self, policy_data, rebase_step):
        base = self.value(policy_data, rebase_step, 'Base')
        mort_scr = self.value(policy_data, rebase_step, 'Mortality') - base
        int_scr = self.value(policy_data, rebase_step, 'Interest') - base
        bscr = sqrt(mort_scr ** 2 + int_scr ** 2 + 0.25 * mort_scr * int_scr)
        return base + bscr





if __name__ == '__main__':
    ann_data = pd.read_csv('game_ann_data.csv')
    ann = Annuity('eiopa_spot_annual', 1)
    base1 = ann.expected_cfs(ann_data)


