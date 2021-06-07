import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from matplotlib.patches import Polygon

class ModelBasedImputation:

    def apply_model_based_imputation(self, data_table, col, info=False):

        # Remove all missing values
        data_no_nan = data_table.dropna(axis=0)

        # split data into features and target
        X = data_no_nan.drop(col, axis=1)
        y = data_no_nan[col]

        # Select rows with missing column instances (boolean values)
        row_idx_missing = data_table[col].isnull()
        # print('1', len(row_idx_missing))
        data_missing = pd.DataFrame(data_table[row_idx_missing])
        # print('2', len(data_missing))
        data_missing.drop(col, axis=1, inplace=True)

        # Fill missing values for features that are not of interest
        col_means = data_missing.mean(axis=0)
        data_missing.fillna(value=col_means, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        lm = LinearRegression().fit(X_train, y_train)

        pred = lm.predict(data_missing)

        data_table.loc[row_idx_missing, col] = pred

        if info:
            scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=10)
            print('Average R2 score:', np.mean(scores))

            temp_data = data_table.copy(deep=True)
            temp_data['impute'] = row_idx_missing

            label_cols = [c for c in temp_data if c.startswith('label')]

            data = []
            for label in label_cols:
                for miss in [False, True]:
                    temp = temp_data.loc[(temp_data[label] == 1) & (temp_data['impute'] == miss)][col].values
                    data.append(temp)

            self.plot_distribution_imputation(data, label_cols)

        return data_table

    def plot_distribution_imputation(self, data, label_cols):
        fig, ax1 = plt.subplots(figsize=(15, 6))
        fig.canvas.manager.set_window_title('A Boxplot Example')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)

        ax1.set(
            axisbelow=True,  # Hide the grid behind plot objects
            title='Model based imputation',
            xlabel='Distribution',
            ylabel='Value',
        )

        # Now fill the boxes with desired colors
        box_colors = ['darkkhaki', 'royalblue']
        num_boxes = len(data)
        medians = np.empty(num_boxes)
        for i in range(num_boxes):
            box = bp['boxes'][i]
            box_x = []
            box_y = []
            for j in range(5):
                box_x.append(box.get_xdata()[j])
                box_y.append(box.get_ydata()[j])
            box_coords = np.column_stack([box_x, box_y])
            # Alternate between Dark Khaki and Royal Blue
            ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
            # Now draw the median lines back over what we just filled in
            med = bp['medians'][i]
            median_x = []
            median_y = []
            for j in range(2):
                median_x.append(med.get_xdata()[j])
                median_y.append(med.get_ydata()[j])
                ax1.plot(median_x, median_y, 'k')
            medians[i] = median_y[0]
            # Finally, overplot the sample averages, with horizontal alignment
            # in the center of each box
            ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                     color='w', marker='*', markeredgecolor='k')

        # Set the axes labels
        ax1.set_xticklabels(np.repeat(label_cols, 2),
                            rotation=45, fontsize=8)

        # Due to the Y-axis scale being different across samples, it can be
        # hard to compare differences in medians across the samples. Add upper
        # X-axis tick labels with the sample medians to aid in comparison
        # (just use two decimal places of precision)
        pos = np.arange(num_boxes) + 1
        upper_labels = [str(round(s, 2)) for s in medians]
        weights = ['bold', 'semibold']
        for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
            k = tick % 2
            ax1.text(pos[tick], .95, upper_labels[tick],
                     transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='x-small',
                     weight=weights[k], color=box_colors[k])

        # Finally, add a basic legend
        fig.text(0.80, 0.08, 'Heart rate original',
                 backgroundcolor=box_colors[0], color='black', weight='roman',
                 size='x-small')
        fig.text(0.80, 0.045, 'Heart rate imputed',
                 backgroundcolor=box_colors[1],
                 color='white', weight='roman', size='x-small')
        fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
                 weight='roman', size='small')
        fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
                 size='x-small')

        plt.show()
