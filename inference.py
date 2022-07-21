import numpy as np

from dataset import DataModule
from model import MayoModel


# 1e-4 transformer epoch 30
# 1e-3 fc          epoch 4
# 1e-2 conv        epoch 7

model_fc_0 = MayoModel.load_from_checkpoint('lightning_logs/fc_fold0/version_0/checkpoints/epoch=16-step=9758.ckpt')
model_fc_0 = model_fc_0.eval()

model_fc_1 = MayoModel.load_from_checkpoint('lightning_logs/fc_fold1/version_0/checkpoints/epoch=15-step=9184.ckpt')
model_fc_1 = model_fc_1.eval()

model_fc_2 = MayoModel.load_from_checkpoint('lightning_logs/fc_fold2/version_0/checkpoints/epoch=18-step=10906.ckpt')
model_fc_2 = model_fc_2.eval()

model_fc_3 = MayoModel.load_from_checkpoint('lightning_logs/fc_fold3/version_0/checkpoints/epoch=17-step=10332.ckpt')
model_fc_3 = model_fc_3.eval()

model_fc_4 = MayoModel.load_from_checkpoint('lightning_logs/fc_fold4/version_0/checkpoints/epoch=20-step=12054.ckpt')
model_fc_4 = model_fc_4.eval()

# model_fc = ScoringModel.load_from_checkpoint('lightning_logs/4models_1e-3/version_0/checkpoints/epoch=4-step=3590.ckpt')
# model_fc.eval()
#
# model_conv = ScoringModel.load_from_checkpoint('lightning_logs/4models_1e-2/version_0/checkpoints/epoch=7-step=5744.ckpt')
# model_conv.eval()


#print('model device ', model.device)

dm = DataModule(train_batch_size=12, eval_batch_size=10000, train_or_test='test')
mean_values = []
weights = []
all_time_series = []
with open('submission.csv', 'a') as f:
    f.write('customer_ID,prediction\n')
    for idx, x in enumerate(dm.test_dataloader()):
        #print('x ', x)
        id_for_submission = x['id']

        # time_series = x['time_series']
        #print('time_series ', time_series.shape)
        # mean_value = np.mean(time_series.cpu().numpy(), axis=(0, 2))
        # mean_values.append(mean_value)
        # weights.append(len(time_series))
        # all_time_series.append(time_series)
        #print('mean_value ', mean_value.shape)
        #input()

        preds_0 = model_fc_0(**x)['out_fc'].data.cpu().numpy()
        preds_1 = model_fc_1(**x)['out_fc'].data.cpu().numpy()
        preds_2 = model_fc_2(**x)['out_fc'].data.cpu().numpy()
        preds_3 = model_fc_3(**x)['out_fc'].data.cpu().numpy()
        preds_4 = model_fc_4(**x)['out_fc'].data.cpu().numpy()
        # preds_fc = model_transformer(**x)['out_fc'].data.cpu().numpy()
        # preds_conv = model_transformer(**x)['out_conv'].data.cpu().numpy()
        preds = (preds_0 + preds_1 + preds_2 + preds_3 + preds_4) / 5.0
        if idx % 1000 == 0:
            print('idx ', idx)
            #print('id_for_submission ', id_for_submission)
            #print('preds ', preds, preds.shape)
        for id, score in zip(id_for_submission, preds):
            # if score >= 0.5:
            #     score = 1.0
            # else:
            #     score = 0.0
            f.write(id + ',' + str(score) + '\n')

# print('all_time_series ', len(all_time_series), all_time_series[0].shape)
# input()
# all_time_series = np.concatenate(all_time_series)
# print('all_time_series ', all_time_series.shape) # all_time_series  (924621, 189, 13)
# input()
#
# mean_time_series = all_time_series.mean(axis=(0, 2))
# print('mean_time_series ', mean_time_series, mean_time_series.shape)
# input()
# maxes = all_time_series.max(axis=(0, 2))
# print('maxes_time_series ', maxes, maxes.shape)
# input()
# mins = all_time_series.min(axis=(0, 2))
# print('mins_time_series ', mins, mins.shape)
# input()
# vars = all_time_series.var(axis=(0, 2))
# print('vars_time_series ', vars, vars.shape)
# input()
#
# np.save('all_means_features_test.npy', maxes)
# np.save('all_maxes_features_test.npy', maxes)
# np.save('all_mins_features_test.npy', mins)
# np.save('all_vars_features_test.npy', vars)
#
#
#
# mean_features = np.zeros(189)
# for m_v, w in zip(mean_values, weights):
#     mean_features += w * m_v
#
# sum_weights = np.array(weights).sum()
# mean_features = mean_features / sum_weights
# #print('mean_features ', mean_features, mean_features.shape)
# np.save('mean_features_test.npy', mean_features)