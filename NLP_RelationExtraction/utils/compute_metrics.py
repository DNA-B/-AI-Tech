import wandb
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

label_list = ['no_relation', 'org:top_members/employees', 'org:members', 'org:product', 'per:title', 'org:alternate_names',
                'per:employee_of', 'org:place_of_headquarters', 'per:product',
                'org:number_of_employees/members', 'per:children',
                'per:place_of_residence', 'per:alternate_names',
                'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                'org:member_of', 'per:parents', 'org:dissolved',
                'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                'per:religion']
  
def viz(labels, preds, probs):
    wandb.log({
            "auprc": wandb.plot.roc_curve(labels, probs, labels=label_list),
            "precision_recall": wandb.plot.pr_curve(labels, probs, labels=label_list),  
            "Confusion Matrix": wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=label_list)
        })

def klue_re_micro_f1(preds, labels):
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)
    
    viz(labels, preds, probs)
    
    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc
    }