import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MOB_PLOT :
        
    @staticmethod
    def plotBinsSummary(monoOptBinTable, bar_fill = 'skyblue', bar_alpha = 0.5, bar_width = 0.5, bar_text_color = 'darkblue', 
                        line_color = 'orange', line_width = 3, dot_color = 'red', dot_size = 80, annotation_font_weight = 'bold', 
                        figsavePath: str = None , dpi:int = 300):
        
        fig, ax1 = plt.subplots(1,1,figsize = (12,8))
        binSummaryTable = monoOptBinTable.copy()
        var_name = binSummaryTable.index.name        
        binSummaryTable['interval'] = '[' + binSummaryTable['[intervalStart'].astype(str) + ' , ' + binSummaryTable['intervalEnd)'].astype(str) + ')'
        
        # Plot bar chart for 'dist_obs'
        bars = ax1.bar(np.arange(len(binSummaryTable['interval'])), binSummaryTable['woe'], color = bar_fill, alpha = bar_alpha, width = bar_width)
        ax1.set_xticks(ticks = np.arange(len(binSummaryTable['interval'])), labels = binSummaryTable['interval'])
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
        ax2.plot(np.arange(len(binSummaryTable['interval'])), binSummaryTable['bad_rate'], color=line_color, label='Bad Rate', linewidth = line_width)
        ax2.scatter(np.arange(len(binSummaryTable['interval'])), binSummaryTable['bad_rate'], color=dot_color, s = dot_size)
        ax2.set_xticks(ticks = np.arange(len(binSummaryTable['interval'])), labels = binSummaryTable['interval'])
        ax2.set_ylabel('Bad Rate', color=dot_color)
        
        # Add text
        med = binSummaryTable['bad_rate'].median()
        if binSummaryTable.iloc[-1, 4] - binSummaryTable.iloc[0, 4] > 0 : # bar_rate
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
            fig.patch.set_facecolor('white')
            plt.savefig(figsavePath, dpi = dpi)
        # Show the plot
        plt.show()

    @staticmethod
    def plotPAVACsd(CSD_Summary, figsavePath: str = None , dpi:int = 300) :
        
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        var = CSD_Summary.columns[0]
        response = CSD_Summary.iloc[:,1].name.split('_')[0]
        metric = CSD_Summary.iloc[:,1].name.split('_')[1]
        
        _GCM = CSD_Summary[['intervalStart', 'intervalEnd', 'assignMetric']].drop_duplicates(['intervalStart', 'intervalEnd', 'assignMetric'])
        _GCM['[intervalStart'] = _GCM['intervalStart'].astype(str)
        _GCM['intervalEnd)'] = _GCM['intervalStart'].shift(-1).astype(str)
        _GCM.iloc[0,3] = str(np.inf)
        _GCM.iloc[-1,4] = str(-np.inf)
        _GCM['interval'] = '[' + _GCM['[intervalStart'] + ',' + _GCM['intervalEnd)'] + ')'
        '''
        GCM
        intervalStart | intervalEnd | assignMetric | [intervalStart | intervalEnd) | interval
        -------------------------------------------------------------------------------------
        '''
        
        _CSD = CSD_Summary.drop_duplicates(var, keep = 'last')
        # _GCM['interval'] = 
        # Stats of PavmonoNode to plot CSD [self.var]
        ax.plot(_CSD.iloc[:, 0], _CSD.iloc[:,1], 'bo-', label='CSD')

        # PAVA result and assignment to to plot GCM ['assignValue']
        # intervalStart | intervalEnd | assignMetric
        ax.plot(_GCM.iloc[:, 1], _GCM.iloc[:,2], 'ro-', label='GCM')

        # Scatter plot for CSD
        ax.scatter(_CSD.iloc[:, 0], _CSD.iloc[:,1], color='blue')

        # Scatter plot for GCM 
        ax.scatter(_GCM.iloc[:, 1], _GCM.iloc[:,2], color='red')

        #Add text
        if _GCM.iloc[-1, 2] - _GCM.iloc[0, 2] > 0: # last metric - first metric -> get the sign : Greater than 0 --> '+'
            for i, val in enumerate(_GCM['intervalEnd']):
                ax.annotate(f'{_GCM.iloc[i, 5]}', xy=(val,_GCM.iloc[i, 2]), xytext=(2, -10), textcoords='offset points', ha='left', va='top', weight = 'bold', c = 'red', size = 9)
        else :
            for i, val in enumerate(_GCM['intervalEnd']):
                ax.annotate(f'{_GCM.iloc[i, 5]}', xy=(val,_GCM.iloc[i, 2]), xytext=(2, -10), textcoords='offset points', ha='left', va='top', weight = 'bold', c = 'red', size = 9)

        plt.xlabel(var)
        plt.ylabel(metric)
        plt.title(f'Pool Adjacent Violators : <{var}, {response}> on "{metric}"')

        plt.legend(loc = 'best')
        
        if figsavePath != None :
            fig.patch.set_facecolor('white')
            plt.savefig(figsavePath, dpi = dpi)
        
        plt.show()
