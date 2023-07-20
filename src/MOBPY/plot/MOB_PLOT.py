import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MOB_PLOT :
        
    @staticmethod
    def plotBinsSummary(monoOptBinTable, var_name, bar_fill = 'skyblue', bar_alpha = 0.5, bar_width = 0.5, bar_text_color = 'darkblue', 
                        line_color = 'orange', line_width = 3, dot_color = 'red', dot_size = 80, annotation_font_weight = 'bold', 
                        figsavePath: str = None , dpi:int = 300):
        
        fig, ax1 = plt.subplots(1,1,figsize = (12,8))
        binSummaryTable = monoOptBinTable.copy()
        binSummaryTable['end'] = pd.Categorical(binSummaryTable['end'])

        # Plot bar chart for 'dist_obs'
        bars = ax1.bar(np.arange(len(binSummaryTable['end'])), binSummaryTable['woe'], color = bar_fill, alpha = bar_alpha, width = bar_width)
        ax1.set_xticks(ticks = np.arange(len(binSummaryTable['end'])), labels = binSummaryTable['end'])
        ax1.axhline(0)
        ax1.set_xlabel('Interval End Value')
        ax1.set_ylabel('WoE', color = bar_text_color)

        # Add text
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height >= 0 :
                ax1.annotate(f'{binSummaryTable["dist_obs"].iloc[i]:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 10), textcoords='offset points', ha='center', va='top', weight = annotation_font_weight, c = bar_text_color)
            else :
                ax1.annotate(f'{binSummaryTable["dist_obs"].iloc[i]:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, -8), textcoords='offset points', ha='center', va='top', weight = annotation_font_weight, c = bar_text_color)
        ax2 = ax1.twinx()

        # Plot line chart for 'bad_rate'
        ax2.plot(np.arange(len(binSummaryTable['end'])), binSummaryTable['bad_rate'], color=line_color, label='Bad Rate', linewidth = line_width)
        ax2.scatter(np.arange(len(binSummaryTable['end'])), binSummaryTable['bad_rate'], color=dot_color, s = dot_size)
        ax2.set_xticks(ticks = np.arange(len(binSummaryTable['end'])), labels = binSummaryTable['end'])
        ax2.set_ylabel('Bad Rate', color=dot_color)
        
        # Add text
        med = binSummaryTable['bad_rate'].median()
        if binSummaryTable.iloc[-1, 4] - binSummaryTable.iloc[0, 4] > 0 :
            for i, val in enumerate(binSummaryTable['bad_rate']):
                if val <= med :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, -7.5), textcoords='offset points', ha='left', va='top', weight = annotation_font_weight, c = dot_color)
                else :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, 7.5), textcoords='offset points', ha='right', va='bottom', weight = annotation_font_weight, c = dot_color)
        else :
            for i, val in enumerate(binSummaryTable['bad_rate']):
                if val <= med :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, -7.5), textcoords='offset points', ha='right', va='top', weight = annotation_font_weight, c = dot_color)
                else :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, 7.5), textcoords='offset points', ha='left', va='bottom', weight = annotation_font_weight, c = dot_color)
        # Set Legend        
        plt.legend(labels = ['Bar Text : obs_dist', 'Dot Text : bad_rate'], loc = 'lower center')
        # Set title
        plt.title(f'Bins Summary Plot - {var_name} \n IV : {(binSummaryTable["iv_grp"].sum()):.4f}')

        if figsavePath != None :
            plt.savefig(figsavePath, dpi = dpi)
        # Show the plot
        plt.show()

    @staticmethod
    def plotPAVA_CSD(CSD_Summary) :
        
        plt.figure(figsize=(12,8))
        var = CSD_Summary.columns[0]
        response = CSD_Summary.iloc[:,1].name.split('_')[0]
        metric = CSD_Summary.iloc[:,1].name.split('_')[1]
        
        # Line 1: x-axis is self.var, y-axis is self.metric, color is blue
        plt.plot(CSD_Summary.iloc[:, 0], CSD_Summary.iloc[:,1], 'bo-', label='CSD')

        # Line 2: x-axis is assignValue, y-axis is assignMetric, color is red
        plt.plot(CSD_Summary.iloc[:, 2], CSD_Summary.iloc[:,3], 'ro-', label='GCM')

        # Scatter plot for Line 1
        plt.scatter(CSD_Summary.iloc[:, 0], CSD_Summary.iloc[:,1], color='blue')

        # Scatter plot for Line 2
        plt.scatter(CSD_Summary.iloc[:, 2], CSD_Summary.iloc[:,3], color='red')

        # Set labels and title
        plt.xlabel(var)
        plt.ylabel(metric)
        plt.title(f'Pool Adjacent Violators : <{var}, {response}> on "{metric}"')

        # Add a legend
        plt.legend(loc = 'best')

        # Display the chart
        plt.show()
