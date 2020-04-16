import tensorflow as tf

from definition import ROOT_DIR




import subprocess


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.
    Returns: None
    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = out.stdout.split(b'\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))
    return sum(scores) / len(scores)


# Read test file line
lines = io.open(source, encoding='UTF-8').read().strip().split('\n')
lines = [s for s in lines]


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


pred_file_path = os.path.join(ROOT_DIR, "prediction")
"""
new_start = time.time()
test_target = ""
with  open(pred_file_path, 'w', encoding='utf-8', buffering=1) as pred_file:
    for (batch, (inp)) in enumerate(test_dataset):
        if batch % 5 == 0:
            print("Evaluating for batch", batch)
        pred_fr = translate_batch(inp, batch_size=64)
        for p_fr in pred_fr:
            pred_file.write(p_fr.strip() + '\n')
"""
# print('Time taken for Evaluation: {} secs\n'.format(time.time() - new_start))
print("Files saved:", pred_file_path)
score = compute_bleu(pred_file_path, test_target, False)
print("Bleu Score: ", score)
print("-------------")
