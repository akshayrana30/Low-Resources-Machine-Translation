# root_path = "/content/drive/My Drive/NMT/"
root_path = "/Users/akshayrana/Documents/Github/Low-Resources-Machine-Translation/Akshay/data/"
unaligned_en_path = "unaligned.en"
unaligned_fr_path = "unaligned.fr"
aligned_en_path = "train.lang1"
aligned_fr_path = "train.lang2"


# Most frequent words will be chosen.
inp_vocab_size=20000
tar_vocab_size=20000

# While keeping an embedding of 128 or 256, the batch size can be increased upto 128. 
emb_size = 256
train_batch_size = 128
val_batch_size = 256


train_val_split_ratio = 0.2
random_seed_for_split = 1234


# This will load word2vec embeddings for vocab size 20k. 
# Only works with emb_size 128 or 256. 
# Might need to create word2vec separately for other embedding dimensions.
if emb_size == 128 or emb_size == 256:
    load_emb = True
else:
    load_emb = False

# Transformers parameters
num_layers = 4 
d_model = 256  
dff = 1024      
num_heads = 8 
dropout_rate = 0.1

# If you need to train from scratch, change this to False
load_from_checkpoint = False
checkpoint_path = root_path+"eng_to_french_checkpoints/train/"

if load_from_checkpoint: # and a checkpoint exists in the path above
    # Train for few epoch if already loading from checkpoint
    EPOCHS = 10
else:
    EPOCHS = 30

# Save checkpoint after these many epochs 
save_every = 5

# Evaluate validation loss after these many epochs.
evaluate_val_loss_every = 5

# Evaluate Bleu after these many epochs. This may take time 
evaluate_bleu_every = 5

# Reverse translate: If true, it will translate from French -> to English.
# Model can be used to generate synthetic samples from monolinguals
reverse_translate=False

# This will add synthetic data loaded at "aligned_en_synth_path" 
add_synthetic_data=True

if reverse_translate:
    # This is to support French -> English model
    # X is predicted, Y is real data..
    # The below data is generated using a english to french model.
    aligned_en_synth_path = "synth_en_fr_input_EN.txt"   # Original English.
    aligned_fr_synth_path = "synth_en_fr_output_FR.txt"  # Predicted French.
else:
    # This is to support English -> French model
    # X is predicted, Y is real data..
    # The below data is generated using a french to english model.
    aligned_en_synth_path = "synth_fr_en_output_EN_MIX.txt" # Predicted English.
    aligned_fr_synth_path = "synth_fr_en_input_FR_MIX.txt"  # Original French.


# This is to generate synthetic parallel samples from monolingual data..
# Works with both types of model to generate iterative back translation
generate_samples = False
number_of_samples = 30000

if reverse_translate:
    # French -> English model will generate English sentences from French monolingual.
    generate_input_path = root_path + "fr_en_input_FR.txt"
    generate_output_path = root_path + "fr_en_output_EN.txt"
else:
    # English -> French model will generate French sentences from English monolingual.
    generate_input_path = root_path + "en_fr_input_EN.txt"
    generate_output_path = root_path + "en_fr_output_FR.txt"