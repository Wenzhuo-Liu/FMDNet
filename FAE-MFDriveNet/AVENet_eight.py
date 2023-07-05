from image_convnet import *
from audio_convnet import *
from incline_convnet import *
from KFX_a_convnet import *
from KFY_a_convnet import *
from KFZ_a_convnet import *
from roll_convnet import *
from Yaw_convnet import *
import math
from utils.mydata_xu import *
## Main NN starts here
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
from Synchronized_BatchNorm_PyTorch.sync_batchnorm import SynchronizedBatchNorm2d

class AVENet(nn.Module):

	def __init__(self):
		super(AVENet, self).__init__()

		self.relu   = F.relu
		self.imgnet = ImageConvNet()
		self.incnet = InclineConvNet()
		self.kfxnet = KFX_aConvNet()
		self.kfynet = KFY_aConvNet()
		self.kfznet = KFZ_aConvNet()
		self.audnet = AudioConvNet()
		self.rolnet = RollConvNet()
		self.yawnet = YawConvNet()

		# Vision subnetwork
		self.vpool4  = nn.MaxPool2d(14, stride=14)
		self.vfc1    = nn.Linear(512, 128)
		self.vfc2    = nn.Linear(128, 128)
		self.vl2norm = nn.BatchNorm1d(128)

		# Incline subnetwork
		self.incpool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.incfc1    = nn.Linear(512, 128)
		self.incfc2    = nn.Linear(128, 128)
		self.incl2norm = nn.BatchNorm1d(128)

		# KFX subnetwork
		self.kfxpool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.kfxfc1    = nn.Linear(512, 128)
		self.kfxfc2    = nn.Linear(128, 128)
		self.kfxl2norm = nn.BatchNorm1d(128)
		
		# KFY subnetwork
		self.kfypool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.kfyfc1    = nn.Linear(512, 128)
		self.kfyfc2    = nn.Linear(128, 128)
		self.kfyl2norm = nn.BatchNorm1d(128)

		# KFZ subnetwork
		self.kfzpool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.kfzfc1    = nn.Linear(512, 128)
		self.kfzfc2    = nn.Linear(128, 128)
		self.kfzl2norm = nn.BatchNorm1d(128)

		# Audio subnetwork
		self.apool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.afc1    = nn.Linear(512, 128)
		self.afc2    = nn.Linear(128, 128)
		self.al2norm = nn.BatchNorm1d(128)
		
		# Roll subnetwork
		self.rolpool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.rolfc1    = nn.Linear(512, 128)
		self.rolfc2    = nn.Linear(128, 128)
		self.roll2norm = nn.BatchNorm1d(128)
		
		# Yaw subnetwork
		self.yawpool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.yawfc1    = nn.Linear(512, 128)
		self.yawfc2    = nn.Linear(128, 128)
		self.yawl2norm = nn.BatchNorm1d(128)

		# Combining layers
		self.mse     = F.mse_loss
		#self.fc3     = nn.Linear(1, 2)
		self.fc3     = nn.Linear(128, 3)
		self.softmax = F.softmax

		#FAE-FM
		self.oneconv512 = nn.Conv2d(1024, 512, 1, stride=1, padding=0)
		self.oneconv256 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
		self.oneconv128 = nn.Conv2d(256, 128, 1, stride=1, padding=0)

		self.oneconv1 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
		

		self.norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)
		self.conv1 = nn.Conv2d(128, 128, 1)
		self.k = 64
		self.linear_0 = nn.Conv1d(128, self.k, 1, bias=False)
		self.linear_1 = nn.Conv1d(self.k, 128, 1, bias=False)
		self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)    
		self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),self.norm_layer(128))  
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, _BatchNorm):
				m.weight.data.fill_(1)
				if m.bias is not None:
					m.bias.data.zero_()




	def forward(self, image, incline, kfx_a, kfy_a, kfz_a, audio, roll, Yaw):
		# Image
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)

		# incline
		inc = self.incnet(incline)
		inc = self.incpool4(inc).squeeze(2).squeeze(2)
		inc = self.relu(self.incfc1(inc))
		inc = self.incfc2(inc)
		inc = self.incl2norm(inc)
		
		# KFX
		kfx = self.kfxnet(kfx_a)
		kfx = self.kfxpool4(kfx).squeeze(2).squeeze(2)
		kfx = self.relu(self.kfxfc1(kfx))
		kfx = self.kfxfc2(kfx)
		kfx = self.kfxl2norm(kfx)

		# KFY
		kfy = self.kfynet(kfy_a)
		kfy = self.kfypool4(kfy).squeeze(2).squeeze(2)
		kfy = self.relu(self.kfyfc1(kfy))
		kfy = self.kfyfc2(kfy)
		kfy = self.kfyl2norm(kfy)
		
		# KFz
		kfz = self.kfznet(kfz_a)
		kfz = self.kfzpool4(kfz).squeeze(2).squeeze(2)
		kfz = self.relu(self.kfzfc1(kfz))
		kfz = self.kfzfc2(kfz)
		kfz = self.kfzl2norm(kfz)

		# Audio
		aud = self.audnet(audio)
		aud = self.apool4(aud).squeeze(2).squeeze(2)
		aud = self.relu(self.afc1(aud))
		aud = self.afc2(aud)
		aud = self.al2norm(aud)
		
		# Roll
		rol = self.rolnet(roll)
		rol = self.rolpool4(rol).squeeze(2).squeeze(2)
		rol = self.relu(self.rolfc1(rol))
		rol = self.rolfc2(rol)
		rol = self.roll2norm(rol)

		# Yaw
		yaw = self.yawnet(Yaw)
		yaw = self.yawpool4(yaw).squeeze(2).squeeze(2)
		yaw = self.relu(self.yawfc1(yaw))
		yaw = self.yawfc2(yaw)
		yaw = self.yawl2norm(yaw)


		# Join them 
		#mse = self.mse(img, aud, reduce=False).mean(1).unsqueeze(1)
		aud=aud.unsqueeze(-1).unsqueeze(-1)
		img=img.unsqueeze(-1).unsqueeze(-1)
		inc=inc.unsqueeze(-1).unsqueeze(-1)
		kfx=kfx.unsqueeze(-1).unsqueeze(-1)
		kfy=kfy.unsqueeze(-1).unsqueeze(-1)
		kfz=kfz.unsqueeze(-1).unsqueeze(-1)
		rol=rol.unsqueeze(-1).unsqueeze(-1)
		yaw=yaw.unsqueeze(-1).unsqueeze(-1)

		# print(aud.shape, img.shape, inc.shape)

		# x=aud+img

		x = torch.cat((aud, img, inc, kfx, kfy, kfz, rol, yaw), dim=1)
		print(x.shape)
		x = self.oneconv512(x)
		x = self.oneconv256(x)
		x = self.oneconv128(x)
		

		# x = x.squeeze(-1).squeeze(-1)
		# x = self.fc128(x)
		# x = x.unsqueeze(-1).unsqueeze(-1)

		idn = x
		x = self.conv1(x)
		b, c, h, w = x.size()
		n = h*w
		x = x.view(b, c, h*w)   # b * c * n 
		attn = self.linear_0(x) # b, k, n
		attn = F.softmax(attn, dim=-1) # b, k, n
		attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
		x = self.linear_1(attn) # b, c, n
		x = x.view(b, c, h, w)
		x = self.conv2(x)

		# x = x + idn
		x = torch.cat((x, idn),1)
		x = self.oneconv1(x)

		x = F.relu(x)
		x=x.squeeze(-1).squeeze(-1)
		#print('x',x.shape)
		out = self.fc3(x)
		#out = self.softmax(out, 1)#对每一行进行softmax

		return out, img, inc, kfx, kfy, kfz, aud, rol, yaw


	def get_image_embeddings(self, image):
		# Just get the image embeddings
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)
		return img

if __name__ == '__main__':
	model = AVENet().cuda()
	
	image = Variable(torch.rand(2, 3, 224, 224)).cuda()
	speed = Variable(torch.rand(2, 1, 257, 200)).cuda()
	inc = Variable(torch.rand(2, 1, 257, 200)).cuda()
	kfx = Variable(torch.rand(2, 1, 257, 200)).cuda()
	kfy = Variable(torch.rand(2, 1, 257, 200)).cuda()
	kfz = Variable(torch.rand(2, 1, 257, 200)).cuda()
	rol = Variable(torch.rand(2, 1, 257, 200)).cuda()
	yaw = Variable(torch.rand(2, 1, 257, 200)).cuda()

	# Run a feedforward and check shape
	o,_,_,_,_,_,_,_,_ = model(image,inc,kfx,kfy,kfz,speed,rol,yaw)

	print(o.shape)#[2,3]
	