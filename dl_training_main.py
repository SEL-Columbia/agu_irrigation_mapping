import numpy as np
import rasterio
import matplotlib.pyplot as plt
import copy

from dl_dataloader import DataGenerator
import tensorflow as tf
from model import DeepClassifier
import argparse, yaml
import datetime
from random import shuffle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib import colors
from tensorflow.keras.models import load_model
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

def get_args():
    parser = argparse.ArgumentParser(
        description="Predicting out of season vegetation growth with a DL model"
    )

    parser.add_argument(
        "--training_params_filename",
        type=str,
        default="params.yaml",
        help="Filename defining model configuration",
    )

    args = parser.parse_args()
    config = yaml.load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args


def calculate_metrics(predictions, true_labels, epoch, summary_string):
    # Calculate loss for summary writing
    loss = tf.keras.losses.binary_crossentropy(true_labels, predictions)

    # Round predictions
    rounded_preds = tf.math.round(predictions)

    rounded_preds = tf.math.argmax(rounded_preds, axis=1)
    true_labels = tf.math.argmax(true_labels, axis=1)

    # acc = tf.math.reduce_mean(tf.math.equal(true_labels, rounded_preds))

    TP = tf.math.count_nonzero(rounded_preds * true_labels)
    TN = tf.math.count_nonzero((rounded_preds - 1) * (true_labels - 1))
    FP = tf.math.count_nonzero(rounded_preds * (true_labels - 1))
    FN = tf.math.count_nonzero((rounded_preds - 1) * true_labels)

    acc = tf.math.divide((TP + TN), (TP + TN + FP + FN))
    precision = tf.math.divide(TP, (TP + FP))
    recall = tf.math.divide(TP, (TP + FN))
    f1 = tf.math.divide(2 * precision * recall, (precision + recall))

    with summary_writer.as_default():
        tf.summary.scalar(f"{summary_string}_loss", tf.math.reduce_mean(loss), step=epoch)
        tf.summary.scalar(f"{summary_string}_acc", acc, step=epoch)
        tf.summary.scalar(f"{summary_string}_precision", precision, step=epoch)
        tf.summary.scalar(f"{summary_string}_recall", recall, step=epoch)
        tf.summary.scalar(f"{summary_string}_f1", f1, step=epoch)

    return TP, TN, FP, FN



@tf.function
def train_step(train_model, features, true_labels, epoch, summary_string, optimizer):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).

        predictions = train_model(features, training=True)
        predictions = tf.squeeze(predictions, axis=1)

        loss = tf.keras.losses.binary_crossentropy(true_labels, predictions)

    gradients = tape.gradient(loss, train_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))

    TP, TN, FP, FN = calculate_metrics(predictions, true_labels, epoch, summary_string)


def predict_over_map(args, ix, t_region, model, generator, test_results, columns_to_use):

    if t_region == 'amhara':
        truth = 'Visual'
    else:
        truth = 'Ground'

    valid_pixels_ds, valid_pixel_indices = generator.return_pixels_for_map_prediction(args, t_region, columns_to_use)
    visual_truth_arrays = generator.visual_truth_arrays
    pixels_within_shapefile_dict = generator.pixels_within_shapefile_dict


    colors_xkcd = ['cobalt', 'very dark purple',  "amber",
                   "faded green",   'terracotta', "pale purple",  'grape',
                   'salmon pink',  'greyish', 'dark turquoise', 'pastel blue'
                   ]

    # sns.set_palette(sns.xkcd_palette(colors_xkcd))


    cmap_im_all_preds = colors.ListedColormap(sns.xkcd_palette(colors_xkcd)[0:4])
    cmap_im_vt = colors.ListedColormap(sns.xkcd_palette(colors_xkcd)[0:6])

    bounds_all_preds = np.arange(start=-2.5, stop=2.5, step=1)
    bounds_vt = np.arange(start=-2.5, stop=4.5, step=1)

    preds_list = []

    print(f'Predict over map for region: {t_region}')
    for features in valid_pixels_ds:
        predictions = model(features, training = False)
        predictions = tf.squeeze(predictions, axis=1)
        preds_list.extend(predictions)

    preds_array = np.array(preds_list)
    preds_array = np.argmax(preds_array, axis=1)



    irrig_vt_map = visual_truth_arrays[f'{t_region}_irrig_vt']
    noirrig_vt_map = visual_truth_arrays[f'{t_region}_noirrig_vt']


    preds_map = np.full(irrig_vt_map.shape, -2)
    preds_map[pixels_within_shapefile_dict[t_region]] = -1



    preds_map[valid_pixel_indices] = preds_array

    fig, ax = plt.subplots(1,2, figsize = (15,6))
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.15)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes('right', size='5%', pad=0.15)

    im0 = ax[0].imshow(preds_map, interpolation='nearest', origin='upper', cmap=cmap_im_all_preds)
    cbar = fig.colorbar(im0, cax=cax0, orientation='vertical', boundaries=bounds_all_preds, ticks=range(-2, 3))
    cbar.ax.set_yticklabels(['N/A', 'No\nPrediction', 'Predicted\nNo Irrigation', 'Predicted\nIrrigation'])

    vt_map = np.full(irrig_vt_map.shape, -2)
    vt_map[pixels_within_shapefile_dict[t_region]] = -1

    pixels_TN = np.where((np.equal(preds_map, 0) * np.equal(noirrig_vt_map, 1)))
    pixels_FN = np.where((np.equal(preds_map, 0) * np.equal(irrig_vt_map, 1)))
    pixels_FP = np.where((np.equal(preds_map, 1) * np.equal(noirrig_vt_map, 1)))
    pixels_TP = np.where((np.equal(preds_map, 1) * np.equal(irrig_vt_map, 1)))

    vt_map[pixels_TN] = 0
    vt_map[pixels_FN] = 1
    vt_map[pixels_FP] = 2
    vt_map[pixels_TP] = 3

    im1 = ax[1].imshow(vt_map, interpolation='nearest', origin='upper', cmap=cmap_im_vt)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical', boundaries=bounds_vt, ticks=range(-2, 5))
    cbar1.ax.set_yticklabels(['N/A', 'No\nLabel', 'True\nNegative', 'False\nNegative',
                              'False\nPositive', 'True\nPositive'])

    region_results = test_results[ix]
    # ax[0].set_title(f'Inference\n')
    ax[1].set_title(f'Comparison with {truth} Truth\n'
                    f'Test set accuracy, irrigated samples: {region_results[0][0]:.3f} ({region_results[0][1]}/{region_results[0][2]})\n'
                    f'Test set accuracy, non-irrigated samples: {region_results[1][0]:.3f} ({region_results[1][1]}/{region_results[1][2]})')

    ax[0].set_title(f'Test set accuracy, irrigated samples: {region_results[0][0]:.3f} ({region_results[0][1]}/{region_results[0][2]})\n'
                    f'Test set accuracy, non-irrigated samples: {region_results[1][0]:.3f} ({region_results[1][1]}/{region_results[1][2]})')

    fig.suptitle(f'{t_region.capitalize()}')

    # Add scale bar
    fontprops = fm.FontProperties(size=12)
    bar_width = 200
    scalebar = AnchoredSizeBar(ax[0].transData,
                               bar_width, '50km', 'lower left',
                               pad=0.3,
                               color='Black',
                               frameon=True,
                               size_vertical=2,
                               fontproperties=fontprops)
    ax[0].add_artist(scalebar)

    plt.tight_layout(pad=1.08, h_pad=1.16, w_pad=None, rect=None)
    plt.savefig(f'output_files/saved_figures/{t_region}_preds.png', dpi=200)
    plt.show()


def model_evaluation(model, val_ds, region, class_type):

    preds_list = []
    true_labels_list = []

    for features, true_labels in val_ds:

        predictions = model(features, training=False)

        preds_list.extend(np.squeeze(predictions.numpy(), axis = 1))
        true_labels_list.extend(true_labels.numpy())

    summary_string = f'validation_{region}_{class_type}'

    TP, TN, FP, FN = calculate_metrics(np.array(preds_list), np.array(true_labels_list), epoch, summary_string)

    class_acc = (TP + TN) /(TP + TN + FP + FN)

    return class_acc.numpy(), (TP + TN).numpy(), (TP + TN + FP + FN).numpy()

def update_model_weights(irrig_acc_list, noirrig_acc_list, train_model, best_model, acc_results_dict, v_regions):

    print(irrig_acc_list)
    print(noirrig_acc_list)

    # Retrieve prior best accuracies
    prior_accs = []
    for v_region in v_regions:
        prior_accs.extend([acc_results_dict[f'{v_region}_irrig'], acc_results_dict[f'{v_region}_noirrig']])

    new_acc = np.sum([[i[0] for i in irrig_acc_list],
                     [j[0] for j in noirrig_acc_list]])


    prior_acc = np.min(prior_accs)


    if len(best_model.get_weights()) == 0:
        best_model = copy.deepcopy(train_model)

    old_train_weights = train_model.get_weights()[-1]
    old_best_weights = best_model.get_weights()[-1]

    if new_acc > prior_acc:
        print('Overall validation accuracy has improved, saving new weights')
        best_model.set_weights(train_model.get_weights())
        for ix, v_region in enumerate(v_regions):
            acc_results_dict[f'{v_region}_irrig'] = irrig_acc_list[ix][0]
            acc_results_dict[f'{v_region}_noirrig'] = noirrig_acc_list[ix][0]
    else:
        print('Overall validation accuracy has not improved, reverting to old weights')
        train_model.set_weights(best_model.get_weights())

    print(np.equal(old_train_weights, train_model.get_weights()[-1]))
    print(np.equal(old_best_weights, best_model.get_weights()[-1]))



    return train_model, best_model, acc_results_dict





if __name__ == '__main__':

    args = get_args()

    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_regions = ['amhara', 'catalonia', 'fresno']
    val_regions = ['amhara', 'catalonia', 'fresno']
    test_regions = [ 'catalonia']

    acc_results_dict = {}
    for v_region in val_regions:
        acc_results_dict[f'{v_region}_irrig'] = 0
        acc_results_dict[f'{v_region}_noirrig'] = 0



    columns = [
               'srtm',

               'evi_annual_corrcoef', 'evi_chirps_corrcoef',
               'evi_at_min_12_chirps_mean',  'evi_at_min_24_chirps_mean',  'evi_at_min_36_chirps_mean',
               'evi_at_min_12_chirps_max', 'evi_at_min_24_chirps_max', 'evi_at_min_36_chirps_max',
               'evi_max_min_ratio_95_5', 'evi_max_min_ratio_90_10', 'evi_max_min_ratio_85_15', 'evi_max_min_ratio_80_20',

               'ndwi_annual_corrcoef', 'ndwi_chirps_corrcoef',
               'ndwi_at_min_12_chirps_mean', 'ndwi_at_min_24_chirps_mean', 'ndwi_at_min_36_chirps_mean',
               'ndwi_at_min_12_chirps_max', 'ndwi_at_min_24_chirps_max', 'ndwi_at_min_36_chirps_max',

               ]

    columns_to_use = [range(len(columns))]

    print([(i, columns[i]) for i in range(len(columns))])

    print('Initializing data generator')
    generator = DataGenerator(args, columns_to_use, dir_time)


    # Load pretrained model
    load_pretrained_model = True
    if load_pretrained_model:
        print('Loading pretrained model')
        model_dir = 'best_trained_all_regions'
        train_model = load_model(f'{args.base_dir}/pretrained_model_files/models/{model_dir}')
        best_model  = load_model(f'{args.base_dir}/pretrained_model_files/models/{model_dir}')

    else:
        print('Train new model')
        # Create two models in order to update weights accoringly
        train_model = DeepClassifier()
        best_model  = DeepClassifier()

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # Define training loss objects
    log_dir = "tensorboard_logs/" + dir_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    lr = 5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    num_epochs = 20
    epoch = 0

    train = False
    if train:
        for epoch in range(num_epochs):
            shuffle(train_regions)
            for t_region in train_regions:
                # Update the training region + find number of iterations to train on all irrig/non-irrig data
                print(f'Training on region {t_region}')
                generator.update_region(t_region)

                optimizer.learning_rate.assign(lr * generator.training_rate_dict[f'{t_region}'])

                count = 0

                for iter in range(generator.num_iters_per_region):
                    train_ds = generator.take_new_batch_for_iter(iter)
                    for features, true_labels in train_ds:
                        count +=1
                        summary_string = f'training_{t_region}'
                        train_step(train_model, features, true_labels, epoch, summary_string, optimizer)

                print(count)

                # Find validation accuracy
                print(f'Validating on remaining regions')

                irrig_acc_list = []
                noirrig_acc_list = []

                # v_regions = [v for v in val_regions if v != t_region]
                for v_region in val_regions:

                    val_irrig_ds, val_noirrig_ds = generator.return_val_or_test_data('val', v_region)


                    print(f'Validating irrigated pixels, region: {v_region}')
                    irrig_acc, irrig_true, irrig_total = model_evaluation(train_model, val_irrig_ds, region=v_region, class_type='irrig')
                    irrig_acc_list.append((irrig_acc, irrig_total))
                    print(f'Validating nonirrigated pixels, region: {v_region}\n')
                    noirrig_acc, noirrig_true, noirrig_total = model_evaluation(train_model, val_noirrig_ds, region=v_region, class_type='no_irrig')
                    noirrig_acc_list.append((noirrig_acc, noirrig_total))

                train_model, best_model, acc_results_dict = update_model_weights(irrig_acc_list, noirrig_acc_list,
                                                           train_model, best_model, acc_results_dict, val_regions)


    print(acc_results_dict)
    #
    # Return test data results
    test_results = []
    for t_region in test_regions:
        test_irrig_ds, test_noirrig_ds = generator.return_val_or_test_data('test', t_region)

        # Calculate irrigated pixels performance
        irrig_acc, irrig_true, irrig_total = model_evaluation(best_model, test_irrig_ds,
                                                              t_region, class_type='irrig')

        # Calculate nonirrigated pixels performance
        noirrig_acc, noirrig_true, noirrig_total = model_evaluation(best_model, test_noirrig_ds,
                                                                    t_region, class_type='no_irrig')

        print(f'Irrigated test set, region {t_region}, accuracy: {irrig_acc} ({irrig_true}/{irrig_total})')
        print(f'Non-irrigated test set, region {t_region}, accuracy: {noirrig_acc} ({noirrig_true}/{noirrig_total})')

        test_results.append(((irrig_acc, irrig_true, irrig_total), (noirrig_acc, noirrig_true, noirrig_total)))

    ## Predict over map
    for ix, t_region in enumerate(test_regions):
        print(t_region)
        predict_over_map(args, ix, t_region, best_model, generator, test_results, columns_to_use)


    save_model = False
    if save_model:
        best_model.save(f'{args.base_dir}/pretrained_model_files/models/{dir_time}')
