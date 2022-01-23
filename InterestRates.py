# Import external libraries
import pandas as pd


class InterestRate:
    def __init__(self, curve_name, steps_py, stress=None):

        self.steps_py = steps_py

        # Read in chosen interest rate curve
        self.base_annual_spot = pd.read_csv(f'inputs/{curve_name}.csv', index_col='Term')

        # Apply chosen stress (if applicable)
        self.base_annual_spot = apply_stress(self.base_annual_spot, stress)

        # Initialise class level cache
        self.cache = {}

    def spot_curve(self, rebase_step=0):

        # Only construct curve if it is not contained in the cache
        if f'spot_curve_{rebase_step}' not in self.cache:

            # If no rebasing applied
            if rebase_step == 0:

                # Construct empty data frame to put curve into. Index of dataframe represents the step at which the
                # rate applies
                df_spot_curve = pd.DataFrame(index=range(1, self.base_annual_spot.index.max() * self.steps_py + 1))

                # Calculate the year corresponding to each step
                df_spot_curve['Year'] = ((df_spot_curve.index - 1) / self.steps_py + 1).astype(int)

                # Get the unprocessed annual spot rate for each step by merging on the year to which the rate is
                # applicable
                df_spot_curve = df_spot_curve.merge(self.base_annual_spot, left_on='Year', right_index=True, how='left')

                # Convert the annual spot rates to rates with the desired step size and cache the result
                self.cache[f'spot_curve_{rebase_step}'] = pd.DataFrame((1 + df_spot_curve['Rate']) ** (1/self.steps_py) - 1, columns=['Rate'])

            # If rebasing is applied
            else:

                # Get non-rebased curve (using code in the if rebase_step==0 block above)
                df_spot_curve = self.spot_curve(0)

                # Discard cuve values prior to the rebase step
                df_spot_curve = df_spot_curve.iloc[rebase_step:]

                # Reindex so that curve index starts at 1 still
                df_spot_curve.index -= rebase_step

                # Cache the resulting curve
                self.cache[f'spot_curve_{rebase_step}'] = df_spot_curve

        # Return cached curve
        return self.cache[f'spot_curve_{rebase_step}']

    def discount_curve(self, rebase_step=0, spread=0):

        # Only construct curve if it is not contained in the cache
        if f'discount_curve_{rebase_step}_{spread}' not in self.cache:

            # Get spot curve as-at rebase step
            df_spot_curve = self.spot_curve(rebase_step=rebase_step)

            # Calculate discount curve from the spot curve, adding a spread if necessary
            df_discount_curve = 1 + spread.to_numpy().reshape(1, -1) + df_spot_curve['Rate'].to_numpy().reshape(-1,1)# ** (-df_spot_curve.index)
            df_discount_curve = pd.DataFrame(df_discount_curve, index=df_spot_curve.index).pow(-df_spot_curve.index, axis=0)

            # Cache the calculated discount curve
            self.cache[f'discount_curve_{rebase_step}_{spread}'] = df_discount_curve

        # Returned cached curve
        return self.cache[f'discount_curve_{rebase_step}_{spread}']

    def forward_curve(self, rebase_step=0):

        # Only construct curve if it is not contained in the cache
        if f'forward_curve_{rebase_step}' not in self.cache:

            # Caclulate the accumulated value of 1 over time by calculating 1 / discount rate
            df_accumulation_factors = 1 / self.discount_curve(rebase_step)

            # First accumulation factor calculated is at time-step 1 so add initial value of 1 at step 0 to accumulated
            # value curve
            df_accumulation_factors.loc[0] = 1
            df_accumulation_factors.sort_index(inplace=True)

            # Get the accumulation factors offset by -1 steps
            df_accumulation_factors['Prev Rate'] = df_accumulation_factors['Rate'].shift(periods=1)

            # Calculate forward rate at step t as the accumulation factor at time t, divided by the accumulation
            # factor at time (t-1), minus 1
            s_forward_curve = df_accumulation_factors['Rate'] / df_accumulation_factors['Prev Rate'] - 1

            # Cache the calculated forward curve
            self.cache[f'forward_curve_{rebase_step}'] = pd.DataFrame(s_forward_curve[1:], columns=['Rate'])

        # Returned cached curve
        return self.cache[f'forward_curve_{rebase_step}']


def apply_stress(base_spot_curve, stress):

    # For SII stress a prescribed spot rate shock curve should be used. The formula for applying this shock curve is
    # {base curve} - {shock curve} * | {base curve} |
    if stress == 'SII':
        sii_spot_rate_stress = pd.read_csv('inputs/sii_interest_rate_stress.csv', index_col='Term')
        base_spot_curve = pd.DataFrame(
            base_spot_curve['Rate'] - sii_spot_rate_stress['Shock'] * base_spot_curve['Rate'].abs(),
            columns=['Rate'])

    # For an AAA interest rate stress we assume that spot rates are zero at all terms
    if stress == 'AAA':
        base_spot_curve['Rate'] = 0

    # For an AA interest rate stress we assume that spot rates are 50% of best estimate spot rates at all terms
    if stress == 'AA':
        base_spot_curve['Rate'] *= 0.5

    return base_spot_curve



if __name__ == '__main__':
    BaseCurves = InterestRate('eiopa_spot_annual', 1, 'AA')
    disc_curve = BaseCurves.forward_curve(10)
    print(disc_curve)