root_path = "/content/drive/My Drive/NMT/"
# root_path = "/Users/akshayrana/Documents/Github/Low-Resources-Machine-Translation/Akshay/data/"
unaligned_en_path = "unaligned.en"
unaligned_fr_path = "unaligned.fr"
aligned_en_path = "train.lang1"
aligned_fr_path = "train.lang2"
aligned_en_synth_path = "synth_fr_en_pred.txt"
aligned_fr_synth_path = "synth_fr_en_true.txt"

inp_vocab_size=20000
tar_vocab_size=20000
reverse_translate=False
add_synthetic_data=True
load_emb = True
emb_size = 256
train_batch_size = 128
val_batch_size = 128

num_layers = 4 #6
d_model = 256  #512
dff = 1024      #2048
num_heads = 8 
dropout_rate = 0.1
load_from_checkpoint = False
checkpoint_path = root_path+"model_with_more_synthetic_data_checkpoints/train/"
EPOCHS = 10

save_every = 5
evaluate_bleu_every = 5
evaluate_val_loss_every = 5