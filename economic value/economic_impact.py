import pandas as pd
import re

df = pd.read_csv('/content/sample_data/time_groups_processed.csv')

maintenance_counts = df.drop_duplicates(['PCUSerialNumber', 'WO_WO#']) \
                      .groupby('PCUSerialNumber') \
                       .size() \
                       .reset_index(name='Past Maintenence')

first_useful_per_wo = df.groupby(['PCUSerialNumber','WO_WO#'])['Useful_Time'].first().reset_index()

total_active_time = first_useful_per_wo.groupby('PCUSerialNumber')['Useful_Time'].sum().reset_index(name='Total Active Time')


unique_combos = df.drop_duplicates(subset=['PCUSerialNumber', 'WO_WO#'])[['PCUSerialNumber', 'WO_WO#']]

result = unique_combos.merge(maintenance_counts, on='PCUSerialNumber') \
                      .merge(total_active_time, on='PCUSerialNumber')

result = result.sort_values(by=['PCUSerialNumber', 'WO_WO#'])


result.to_csv("economic_impact.csv", index=False)


economic = pd.read_csv('/content/sample_data/economic_impact.csv')
matched = pd.read_csv('/content/sample_data/component_fail_report.csv')

matched['Asset_Serial'] = matched['Asset_Serial'].astype(str)
economic['PCUSerialNumber'] = economic['PCUSerialNumber'].astype(str)


merged_df = pd.merge(
    matched,
    economic,
    how='inner',
    left_on=['Asset_Serial', 'WO_WO#'],
    right_on=['PCUSerialNumber', 'WO_WO#']
)

merged_df.drop(columns=['PCUSerialNumber'], inplace=True)

merged_df = merged_df[['Asset_Serial', 'WO_WO#', 'Past Maintenance', 'Total Active Time', 'Matched_Components']]

merged_df = merged_df[merged_df['Matched_Components'] != '[]']

merged_df.to_csv('final_economic_impact.csv', index=False)