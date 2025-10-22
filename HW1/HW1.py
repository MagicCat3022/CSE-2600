import sys
from ISLP import load_data
import matplotlib.pyplot as plt

Hitters = load_data('Hitters')

def Q2_c():
    for col in Hitters:
        if Hitters[col].dtype == 'float64' or Hitters[col].dtype == 'int64':
            print(col)
            print(Hitters[col].median())
            
def Q2_d():
    walks = Hitters['Walks'].to_list()
    walks = sum(walks)
    Hits = Hitters['Hits'].to_list()
    Hits = sum(Hits)
    AtBats = Hitters['AtBat'].to_list()
    AtBats = sum(AtBats)
    neither = AtBats - (walks + Hits)
    data = [walks, Hits, neither]
    plt.pie(data, labels = ['Walks', 'Hits', 'Neither'], autopct='%1.1f%%', )
    plt.show()
    
def Q2_e():
    Hitters['Salary'].fillna(-1, inplace=True)
    SalaryList = Hitters['Salary'].to_list()
    totalNaN = SalaryList.count(-1)
    Hitters.sort_values(by = 'Salary', inplace = True)
    
    NoSalaryDF = Hitters.iloc[0:totalNaN]
    SalaryDF = Hitters.iloc[totalNaN:]
    
    print(f'Total NaN values in Salary column: {totalNaN}')
    
    for col in NoSalaryDF:
        if NoSalaryDF[col].dtype == 'float64' or NoSalaryDF[col].dtype == 'int64':
            print(f"Column: {col}")
            print(f"No Salary Mean: {NoSalaryDF[col].mean()}")
            print(f"Salary Mean: {SalaryDF[col].mean()}")
            print('---')


def Q2_f():
    Hitters['Salary'].fillna(-1, inplace=True)
    SalaryData = Hitters['Salary'].to_list()
    SalaryData = [i for i in SalaryData if i != -1]
    plt.hist(SalaryData, bins = 50, edgecolor = 'black')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.title('Salary histogram')
    plt.show()
    
def Q2_g():
    metrics = ['CAtBat', 'CHits', 'CHmRun', 'CRuns', 'CRBI', 'CWalks']
    Hitters['Salary'].fillna(-1, inplace=True)
    SalaryData = Hitters['Salary']
    SalaryData = SalaryData[SalaryData != -1]
    for metric in metrics:
        metric_data = Hitters.loc[SalaryData.index, metric]
        plt.scatter(metric_data, SalaryData)
        plt.xlabel(metric)
        plt.ylabel('Salary')
        plt.title(f'{metric} vs Salary')
        plt.show()

def Q2_h():
    divisions = Hitters['Division'].unique()
    Hitters.fillna({'Salary': -1}, inplace=True)
    for division in divisions:
        division_data = Hitters[Hitters['Division'] == division]
        salary_data = division_data['Salary']
        salary_data = salary_data[salary_data != -1]
        print(f'Division: {division}, Total Salary: {salary_data.sum():.2f}')
        
        
def Q2_i():
    Hitters.fillna({'Salary': -1}, inplace=True)
    SalaryData = Hitters['Salary']
    SalaryData = SalaryData[SalaryData != -1]
    
    Years12 = SalaryData[Hitters['Years'] > 12]
    Years5 = SalaryData[Hitters['Years'] <= 5]

    plt.boxplot(Years12)
    plt.ylabel('Salary')
    plt.xticks([1], ['>12 Years'])
    plt.title('Salary Distribution by Years of Experience')
    plt.show()
    
    plt.boxplot(Years5)
    plt.ylabel('Salary')
    plt.xticks([1], ['<=5 Years'])
    plt.title('Salary Distribution by Years of Experience')
    plt.show()

def Q2_j():
    for row in Hitters.itertuples():
        batting_avg = row.CHits / row.CAtBat if row.CAtBat != 0 else 0
        Hitters.at[row.Index, 'BattingAvg'] = batting_avg
        
    Hitters.sort_values(by = 'BattingAvg', ascending = False, inplace = True)
    print(Hitters[['League', 'Division', 'Salary', 'BattingAvg']].head(10))
    
def Q2_k():
    for row in Hitters.itertuples():
        batting_avg = row.CHits / row.CAtBat if row.CAtBat != 0 else 0
        Hitters.at[row.Index, 'BattingAvg'] = batting_avg
        
    Hitters.sort_values(by = 'BattingAvg', ascending = False, inplace = True)
    print(Hitters[['CAtBat', 'CHits', 'CHmRun', 'CRBI', 'CWalks', 'Years', 'BattingAvg']].head(5))

Q2_j()
Q2_k()