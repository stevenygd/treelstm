--[[

  Testing script

--]]

require('..')

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments : TODO
local args = lapp [[
Evaluation script using a pretrained model.
  <model_file>  (string)             Pretrained model file path.
  -m,--model    (default dependency) Model architecture: [dependency, constituency, lstm, bilstm]
  -d,--data     (default tmp/)       Testing data path.
  -o,--output   (string)             Output file name.
]]

for k,v in pairs(args) do
    print(k,v)
end

local model_name, model_class
if args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMSim
elseif args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'
  model_class = treelstm.TreeLSTMSim
elseif args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = treelstm.LSTMSim
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = treelstm.LSTMSim
end
local model_structure = args.model
header(model_name .. ' for Semantic Relatedness')

-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
printf('loading testing : %s\n', (data_dir .. args.data))
local test_dir = data_dir .. args.data
local constituency = (args.model == 'constituency')
local test_dataset = treelstm.read_relatedness_dataset(test_dir, vocab, constituency)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local model = model_class.load(args.model_file)

-- print information
header('model configuration')
model:print_config()

-- evaluate
header('Evaluating on test set')
local test_predictions = model:predict_dataset(test_dataset)
local test_score = pearson(test_predictions, test_dataset.labels)
printf('-- test score: %.4f\n', test_score)

-- create predictions directory if necessary
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end

-- get paths
local file_idx = 1
local predictions_save_path

if args.output ~= nil then
  predictions_save_path = string.format(treelstm.predictions_dir .. '/%s', args.output)
else
  while true do
    -- printf("%d" , model.mem_dim)
    -- printf("%d" , model.num_layers) -- This won't be known
    predictions_save_path = string.format(
      treelstm.predictions_dir .. '/rel-%s.eval.%dd.%d.pred', args.model, model.mem_dim, file_idx)
    if lfs.attributes(predictions_save_path) == nil then
      break
    end
    file_idx = file_idx + 1
  end
end

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
for i = 1, test_predictions:size(1) do
  predictions_file:writeFloat(test_predictions[i])
end
predictions_file:close()

-- to load a saved model
-- local loaded = model_class.load(model_save_path)
