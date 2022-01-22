import pandas as pd
import numpy as np
from Mortality import Mortality
from InterestRates import InterestRate
from math import sqrt, exp
from scipy import stats


class BusinessOperations:
    @staticmethod
    def write_annuity_business(num_pols, incl_deferreds=False, first_pol_index=1, steps_py=12):
        policy_id = range(first_pol_index, first_pol_index + num_pols)
        if incl_deferreds:
            min_age = 50
        else:
            min_age = 65
        ph_age_months = np.random.randint(size=num_pols, low=min_age * steps_py, high=95 * steps_py)
        annual_amt = np.random.randint(size=num_pols, low=50, high=1000) * 100
        df_nb_pols = pd.DataFrame()
        df_nb_pols['Policy ID'] = policy_id
        df_nb_pols['Age'] = ph_age_months
        df_nb_pols['Annual Amt'] = annual_amt
        return df_nb_pols


    @staticmethod
    def write_ltm_business(num_pols, first_pol_index=1, steps_py=12):
        df_ltm_nb = pd.DataFrame(index=range(0, num_pols))
        df_ltm_nb['AER'] = np.random.randint(low=50, high=80, size=num_pols) / 1000
        df_ltm_nb['Property Value'] = np.random.randint(low=200, high=800, size=num_pols) * 1000
        df_ltm_nb['LTV'] = np.minimum(1, np.random.normal(0.317488834119463, sqrt(0.00821015556601078), num_pols))
        df_ltm_nb['Loan Value'] = df_ltm_nb['Property Value'] * df_ltm_nb['LTV']
        df_ltm_nb['Age'] = np.random.randint(low=60*steps_py, high=80*steps_py, size=num_pols)
        df_ltm_nb['Spread'] = 0
        return df_ltm_nb[['Age', 'Loan Value', 'Property Value', 'AER', 'Spread']]


    # Rating proportions control what proportion of bonds purchased are rated [GOV, AAA, AA, A, BBB, BB]
    # TODO: Rating nominal proportions not used and all bonds have same (non-random) nominal
    @staticmethod
    def buy_bonds(nominal, first_bond_index=1, rating_proportions=[0.05, 0.07, 0.15, 0.27, 0.44, 0.02], steps_py=12):
        num_bonds_purchased = int(nominal / 5000000)
        df_bonds_purchased = pd.DataFrame()
        df_bonds_purchased['ISIN'] = np.random.randint(size=num_bonds_purchased, low=100000, high=999999)
        df_bonds_purchased['Rating Stat'] = np.random.rand(num_bonds_purchased)

        df_bonds_purchased['Rating'] = np.repeat("BB", num_bonds_purchased)
        df_bonds_purchased.loc[df_bonds_purchased['Rating Stat'] <= 0.984, 'Rating'] = 'BBB'
        df_bonds_purchased.loc[df_bonds_purchased['Rating Stat'] <= 0.581, 'Rating'] = 'A'
        df_bonds_purchased.loc[df_bonds_purchased['Rating Stat'] <= 0.237, 'Rating'] = 'AA'
        df_bonds_purchased.loc[df_bonds_purchased['Rating Stat'] <= 0.064, 'Rating'] = 'AAA'
        df_bonds_purchased.loc[df_bonds_purchased['Rating Stat'] <= 0.006, 'Rating'] = 'GOV'


        df_bonds_purchased['Annual Coupon Rate'] = np.random.randint(size=num_bonds_purchased, low=170, high = 330) / 10000
        df_bonds_purchased.loc[df_bonds_purchased['Rating'] == 'BBB', 'Annual Coupon Rate'] = np.random.randint(size=sum(df_bonds_purchased['Rating'] == 'BBB'), low=110, high = 320) / 10000
        df_bonds_purchased.loc[df_bonds_purchased['Rating'] == 'A', 'Annual Coupon Rate'] = np.random.randint(size=sum(df_bonds_purchased['Rating'] == 'A'), low=90, high = 150) / 10000
        df_bonds_purchased.loc[df_bonds_purchased['Rating'] == 'AA', 'Annual Coupon Rate'] = np.random.randint(size=sum(df_bonds_purchased['Rating'] == 'AA'), low=70, high = 100) / 10000
        df_bonds_purchased.loc[df_bonds_purchased['Rating'] == 'AAA', 'Annual Coupon Rate'] = np.random.randint(size=sum(df_bonds_purchased['Rating'] == 'AAA'), low=10, high = 60) / 10000
        df_bonds_purchased.loc[df_bonds_purchased['Rating'] == 'GOV', 'Annual Coupon Rate'] = np.random.randint(size=sum(df_bonds_purchased['Rating'] == 'GOV'), low=0, high = 10) / 10000

        # TODO: Assuming int rates are flat 4% when simulating coupon rates
        df_bonds_purchased['Annual Coupon Rate'] += 0.04

        df_bonds_purchased['Nominal'] = np.random.randint(size=num_bonds_purchased, low=int(0.5 * nominal / num_bonds_purchased), high=int(1.5 * nominal / num_bonds_purchased))

        df_bonds_purchased['First Coupon Step'] = np.random.randint(size=num_bonds_purchased, low=0, high=12)

        return df_bonds_purchased







if __name__ == '__main__':
    x = BusinessOperations.buy_bonds(249000000)
    print(x)
