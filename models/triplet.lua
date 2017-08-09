
model1 = nn.Sequential()
model1:add(cudnn.SpatialConvolution(1, 64, 3, 3))
model1:add(cudnn.ReLU(true))

model1:add(cudnn.SpatialConvolution(64, 64, 3, 3))
model1:add(cudnn.ReLU(true))

model1:add(cudnn.SpatialConvolution(64, 64, 3, 3))
model1:add(cudnn.ReLU(true))

model1:add(cudnn.SpatialConvolution(64, 64, 3, 3))
model1:add(cudnn.ReLU(true))
--
model1:add(nn.Squeeze())
model1:add(nn.Normalize(2))

--clone the other two networks in the triplet
model2 = model1:clone('weight', 'bias','gradWeight','gradBias')
model3 = model1:clone('weight', 'bias','gradWeight','gradBias')

-- add them to a parallel table
prl = nn.ParallelTable()
prl:add(model1)
prl:add(model2)
prl:add(model3)
prl:cuda()

net = nn.Sequential()
net:add(prl) -- module 1

cc = nn.ConcatTable()

-- feats 1 with 2
cnn_left = nn.Sequential()
cnnpos_dist = nn.ConcatTable()
cnnpos_dist:add(nn.SelectTable(1))
cnnpos_dist:add(nn.SelectTable(2))
cnn_left:add(cnnpos_dist)
cnn_left:add(nn.DotProduct())
cnn_left:add(nn.View(batch_size,1))
cnn_left:cuda()
cc:add(cnn_left)

-- feats 1 with 3
cnn_right = nn.Sequential()
cnnneg_dist = nn.ConcatTable()
cnnneg_dist:add(nn.SelectTable(1))
cnnneg_dist:add(nn.SelectTable(3))
cnn_right:add(cnnneg_dist)
cnn_right:add(nn.DotProduct())
cnn_right:add(nn.View(batch_size,1))
cnn_right:cuda()
cc:add(cnn_right)
cc:cuda()

net:add(cc) -- module 2

net:add(nn.JoinTable(2))
net:cuda()
