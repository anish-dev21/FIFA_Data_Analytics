import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(df, position, ft1, ft2):
    sns.pairplot(df[[ft1, ft2]], diag_kind='kde')
    plt.suptitle(f'Pairplot of {position} Player Features ({ft1} vs. {ft2})', y=1.02)
    plt.show()

# correlation_matrix(df_off, 'Offensive', 'Apps', 'Minutes played')
# correlation_matrix(df_def, 'Defensive', 'Apps', 'Minutes played')
# correlation_matrix(df_mid, 'Midfield', 'Apps', 'Minutes played')