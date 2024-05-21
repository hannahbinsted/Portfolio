# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:33:48 2024

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# import data

data = pd.read_csv('data/covid_data.csv')


# clean data by ensuring date is in datetime format

data['date'] = pd.to_datetime(data['date'])


# cases over time per continent

continent_cases = data.groupby(['date', 'continent'])['total_cases'].sum()
continent_cases = continent_cases.unstack(fill_value=0)

continent_cases.plot(kind='line', figsize=(12,6))

plt.title('Cases per Continent over Time')
plt.xlabel('Date')
plt.ylabel('Cases (in Millions)')

plt.legend(title='Continent')
plt.show()



# deaths over time per continent

continent_deaths = data.groupby(['date', 'continent'])['total_deaths'].sum()
continent_deaths = continent_deaths.unstack(fill_value=0)

continent_deaths.plot(kind='line', figsize=(12,6))

plt.title('Deaths per Continent over Time')
plt.xlabel('Date')
plt.ylabel('Deaths')

plt.legend(title='Continent')
plt.show()



# heatmap of relevant data

first_july_data = data[data['date'] == '2020-07-01']
correlation_matrix = first_july_data[['total_cases_per_million', 'total_deaths_per_million', 'total_tests_per_thousand', 'stringency_index', 'population_density','median_age', 'gdp_per_capita', 'extreme_poverty', 'cvd_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy' ]].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=False, cmap='vlag').set_title('Heatmap of the correlation matrix')



# import income level data from 2020
income_levels = pd.read_csv('data/country_income_levels.csv')
income_levels['Country Code'] = income_levels['Country Code'].str.replace('"', '', regex=True)
income_levels = income_levels.set_index('Country Code')

first_july_data_indexed = first_july_data.set_index('iso_code')

first_july_income_data = pd.concat([first_july_data_indexed, income_levels['IncomeGroup']], axis=1).reindex(first_july_data_indexed.index)



# create colour palette dictionary

my_palette = {'Low income': 'Red', 'Lower middle income': 'Orange', 'Upper middle income': 'Yellow', 'High income': 'Green'}



# scatter plot of cases against tests with regression model

scatter1 = sns.lmplot(data=first_july_income_data, x='total_cases_per_million', y='total_tests_per_thousand', hue='IncomeGroup',palette=my_palette, markers=['+', 'x', 'o', '*'])
scatter1.set(ylim=(0, 350), xlim=(0, 15000), title='Deaths per thousand people against cases per million')



# scatter plot of tests against death rate with regression model

scatter2 = sns.lmplot(data=first_july_income_data, x='cvd_death_rate', y='total_tests_per_thousand', hue='IncomeGroup', palette=my_palette, markers=['+', 'x', 'o', '*'])
scatter2.set(ylim=(0, 200), title='Tests per thousand people against the Covid-19 death rate')



# scatter plot of death rate against stringency index with regression model

scatter3 = sns.lmplot(data=first_july_income_data, y='cvd_death_rate', x='stringency_index', hue='IncomeGroup', palette=my_palette, markers=['+', 'x', 'o', '*'])
scatter3.set(ylim=(50,500), title='Covid-19 death rate against the stringency index')
