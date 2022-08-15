from dataset import DataModule
from model import MayoModel

model = MayoModel.load_from_checkpoint(
    'lightning_logs/downscaled_resnet50_aug/version_8/checkpoints/epoch=8-step=162.ckpt')
model = model.eval()
model.cuda()
print('model device ', model.device)
input()

dm = DataModule(train_batch_size=1, eval_batch_size=1, train_or_test='test')
mean_values = []
weights = []
all_time_series = []
submission_dict = {}

for idx, x in enumerate(dm.test_dataloader()):
    # print('x ', x)
    id_for_submission = x['patient_id'][0]

    preds = model(**x)['out_cls'].data.cpu().numpy()[0]

    if idx % 1 == 0:
        print('idx ', idx)
        print('id_for_submission ', id_for_submission)
        print('preds ', preds, preds.shape)
    if id_for_submission not in submission_dict.keys():
        submission_dict[id_for_submission] = preds
    else:
        print('average predictions!')
        submission_dict[id_for_submission] = (submission_dict[id_for_submission] + preds) * 0.5

print('submission_dict ', submission_dict)
with open('submission.csv', 'a') as f:
    f.write('patient_id,CE,LAA\n')
    for id, score in submission_dict.items():
        f.write(id + ',' + str(score[0]) + ',' + str(score[1]) + '\n')
