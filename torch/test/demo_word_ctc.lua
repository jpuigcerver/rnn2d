require 'rnn2d'
require 'image'
require 'warp_ctc'

-- Decode function
local function decode(output, mapping)
  local W, N, D = output:size(1), output:size(2), output:size(3)
  local _, idx = torch.max(output, 3)
  idx = idx:float()

  local dec = {}
  for n=1,N do
    local dec_n = {}
    for t=1,W do
      if t == 1 or idx[{t, n}][1] ~= idx[{t - 1, n}][1] then
	table.insert(dec_n, idx[{t, n}][1] - 1)
      end
    end
    local txt = ''
    for i=1,#dec_n do
      if dec_n[i] ~= 0 then
	txt = txt .. mapping[dec_n[i]]
      end
    end
    table.insert(dec, txt)
  end
  return dec
end

-- Load input images!
local img1 = image.load('../../assets/labour.png')   -- 1 x H x W
local img2 = image.load('../../assets/tomorrow.png') -- 1 x H x W
img1 = image.scale(img1, img1:size(3) / 8, img1:size(2) / 8)
img2 = image.scale(img2, img2:size(3) / 8, img2:size(2) / 8)

-- Prepare ground-truth
local labels = {'a', 'b', 'l', 'm', 'o', 'r', 't', 'u', 'w'}
local gt = {
  {3, 1, 2, 5, 8, 6},        -- l a b o u r
  {7, 5, 4, 5, 6, 6, 5, 9}   -- t o m o r r o w
}

local K = math.max(img1:size(1), img2:size(1))
local H = math.max(img1:size(2), img2:size(2))
local W = math.max(img1:size(3), img2:size(3))
local N = 2

local img = torch.zeros(N, K, H, W)
img:sub(1, 1,
	1, img1:size(1),
	1, img1:size(2),
	1, img1:size(3)):copy(1.0 - img1)
img:sub(2, 2,
	1, img2:size(1),
	1, img2:size(2),
	1, img2:size(3)):copy(1.0 - img2)
--img = img:permute(3, 4, 1, 2):contiguous()

local model = nn.Sequential()
model:add(rnn2d.LSTM(K     , 10))       -- output shape: N x (4 * D) x H x W
model:add(rnn2d.LSTM(10 * 4, 10))       -- output shape: N x (4 * D) x H x W
model:add(rnn2d.Collapse('sum', 2, 10)) -- output shape: N x D x H x W
model:add(nn.Sum(3))                    -- output shape: N x D x W
model:add(nn.Transpose({1, 2}, {1, 3})) -- output shape: W x N x D
model:add(nn.View(-1, 10))

-- Choose the appropiate backend
-- model = model:cuda()
-- model = model:type('torch.CudaDoubleTensor')
-- model = model:float()
model = model:double()
model:training()
img = img:type(model:type())

param, gradParam = model:getParameters()
start = os.time()
for i=1,10000 do
  y = model:forward(img)
  local gy = y:clone():zero():float()
  losses = cpu_ctc(y:float(), gy, gt, {W, W})
  gy = gy:type(model:type())
  loss = (losses[1] + losses[2])

  model:zeroGradParameters()
  model:backward(img, gy)
  param:add(-0.0005, gradParam)
  if (i - 1) % 200 == 0 or i == 10000 then
    print(string.format('ITER = %04d %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %8d',
			i - 1, y:sum(), y:mean(), y:std(), y:min(), y:max(),
			loss, os.difftime(os.time(), start)))
    print(decode(y:view(W, N, 10), labels))
  end
end
