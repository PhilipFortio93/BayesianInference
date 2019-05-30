import numpy
import time
from matplotlib import pyplot as plt
print ("start 10")

def PowerSpectrumGenerator(map): #reconstructs the power spectrum of the reconstructed map
	##############################################
	#Generate Bins
	#############################################
	binsize = 1
	Bins = [int(i/binsize) for i in range(int(gridwidth*gridwidth/2))]
	countk=[0 for i in range(int(gridwidth*gridwidth*0.5))]
	magkone =[0 for i in range(int(gridwidth*gridwidth*0.5))] # magnitude of the wave vector
	ReconPowerSpectrum = [0 for i in range(int(gridwidth*gridwidth*0.5))] # Reconstructed PowerSpectrum of signal
	n = 0
	m = 0
	while n < gridwidth:
		while m < gridwidth:
			k = (float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
			k=k-1 #shifts k forwards for easy maths
			k=int(k)
			k = Bins[k]
			ReconPowerSpectrum[k] = (ReconPowerSpectrum[k]*countk[k]+numpy.abs(map[n][m]*pixelwidth))/(countk[k]+1)
			countk[k] = countk[k]+1
			m = m + 1
		n = n + 1
		m = 0

	i=0
	while i < int(gridwidth*gridwidth*0.5): #create powerspectrum list for values along diagonal in k space
		magkone[i]=i
		i=i+1

	ReconPowerSpectrum.insert(0, 0) #shifts k back so that the k=0 => S(k)=0
	L=len(ReconPowerSpectrum)-1
	ReconPowerSpectrum.pop(L)
	countk.insert(0, 0) #defines the 0 frequency mode
	L=len(countk)-1
	countk.pop(L)

	i=1
	while i < len(countk): #delete empty values from the spectrum
		if countk[i] == 0:
			countk.pop(i)
			ReconPowerSpectrum.pop(i)
			magkone.pop(i)
			i=i-1
		i=i+1
	
	newishlist=[numpy.sqrt(x) for x in magkone]
	magkone=newishlist
	return magkone, ReconPowerSpectrum #return k and Powerspectrum(k)

####################################
# define the required power spectrum
####################################
def powerspectrumfunc(n): #defines the power spectrum at the nth diag in k space
	sigmasignal = 5
	function=numpy.exp(-float(n)*float(n)/(2*sigmasignal*sigmasignal))/(numpy.sqrt(2*numpy.pi)*(sigmasignal))
	# this function is in fact the square of the pwoer spectrum
	if(function < 0):
		print ("PANIC")
	if(function ==0):
		function = 5.03739653754e-316
	return function

def phasefactor(input): #solves the (-1)^n phase factor input
	n=0
	m=0
	while n < gridwidth:
		while m < gridwidth:
			input[n][m] = input[n][m]*numpy.power(-1,n+m)
			m=m+1
		n=n+1
		m=0
	return input

def WienerMessengerField(N,Sft,datavector,iterations): #Function that returns the reconstructed map
	print ("Started Messenger Field Function")
	tau=0.9 #note lambda = 1.0 implicitely
	t = [0 for i in range(gridwidth*gridwidth)]
	s = [0 for i in range(gridwidth*gridwidth)]
	twoDt = [[0]*gridwidth for i in range(gridwidth)] #noise covariance
	twoDs = [[0]*gridwidth for i in range(gridwidth)] #signal covariance
	Nbarinv=[0 for i in range (gridwidth*gridwidth)]
	T=[sigmanoise*sigmanoise*tau for i in range (gridwidth*gridwidth)]
	Tinv=[0 for i in range (gridwidth*gridwidth)]
	Tftinv=[0 for i in range (gridwidth*gridwidth)]
	Sftinv=[0for i in range (gridwidth*gridwidth)]
	RHS=[0 for i in range (gridwidth*gridwidth)]
	RHS2=[0 for i in range (gridwidth*gridwidth)]
	i=0
	max=len(Nbarinv)

	while i <max: #generate the Nbar=N-T and T in the Fourier domain
		a=float(N[i])
		b=float(T[i])
		c=float(Sft[i])
		Nbarinv[i]=float(float(1)/(a-b))
		Tinv[i]=float(float(1)/b)
		Tftinv[i]=b*(gridwidth*gridwidth)
		Sftinv[i] = float(float(1)/c)
		i=i+1

	k = 0
	while k < iterations:

		start = time.clock()

		i = 0
		while i < max: #calculate t in the pixel domain
			RHS[i] = Nbarinv[i]*datavector[i] + Tinv[i]*s[i]
			t[i] = RHS[i]*(1.0/(Nbarinv[i]+Tinv[i]))
			i = i + 1

		i = 0
		j= 0
		while i < gridwidth: #wrap t into a matrix
			while j < gridwidth:
				twoDt[i][j]= t[i*gridwidth+j]
				j = j + 1
			i = i + 1
			j = 0

		middle1 = time.clock() #this function gives timing for each iteration of the messenger field
		print ("For iteration ", i, " Dot Product took ", middle1-start, " seconds")
		twoDt = numpy.fft.fft2(twoDt) #Fourier transform t
		middle2 = time.clock()
		print ("For iteration ", i, " FT took ", middle2-middle1, " seconds")
		i = 0
		j= 0
		while i < gridwidth: #unwrap FT(t)
			while j < gridwidth:
				t[i*gridwidth+j]= twoDt[i][j]
				j = j + 1
			i = i + 1
			j = 0

		i = 0
		while i < max: #calculate s int he Fourier domain
			RHS2[i] = Tftinv[i]*t[i]
			s[i] = RHS2[i]*(1.0/(Sftinv[i]+Tftinv[i]))
			i = i + 1

		i = 0
		j= 0
		while i < gridwidth:
			while j < gridwidth:
				twoDs[i][j]= s[i*gridwidth+j]
				j = j + 1
			i = i + 1
			j = 0

		middle3 = time.clock()
		
		print ("For iteration ", i, " Dot Product 2 took ", middle3-middle2, " seconds")
		twoDs = numpy.fft.ifft2(twoDs)
		middle4 = time.clock()
		print ("For iteration ", i, " FT2 Took ", middle4 - middle3, " seconds")

		i = 0
		j= 0
		while i < gridwidth:
			while j < gridwidth:
				s[i*gridwidth+j]= twoDs[i][j]
				j = j + 1
			i = i + 1
			j = 0

		print (str(k) + " done")
		k = k + 1
		print middle1-start, middle2-middle1, middle3-middle2,middle4 - middle3

	return s

#######################################
#define the arrays and lengths
#######################################
pixelwidth = 1.0 # width and length of each pixel
global gridwidth
gridwidth=128# make 2^n for int n: this is the edge length of the map
sigmanoise=1# noise variance
griddiag = int(gridwidth*gridwidth*0.5)
map = [[0]*gridwidth for i in range(gridwidth)] #Generated data map
Wiener= [[0]*gridwidth for i in range(gridwidth)]#Wienered map using standard function
powermap= [[0]*gridwidth for i in range(gridwidth)]#powerspectrum in 2D map
magk =[0 for i in range(griddiag)] # magnitude of the wave vector
PowerSpectrum =[0 for i in range(griddiag)] # PowerSpectrum of signal
map = [[0]*gridwidth for i in range(gridwidth)] # map (signal + noise) in the pixel space
signalfreq = [[0]*gridwidth for i in range(gridwidth)] # co-efficients of the frequency domain
countk=[0 for i in range(int(gridwidth*gridwidth*0.5))]
ReconPowerSpectrum = [0 for i in range(int(gridwidth*gridwidth*0.5))] # Reconstructed PowerSpectrum of signal
xaxis = [0 for i in range(gridwidth*gridwidth)]
yaxis = [0 for i in range(gridwidth*gridwidth)]
yaxis2 = [0 for i in range(gridwidth*gridwidth)]
print ("DONE INITIALISING 10")
i=0
while i < griddiag: #create powerspectrum list for values along diagonal in k space
	magk[i]=i
	PowerSpectrum[i] = powerspectrumfunc(i*pixelwidth)
	i=i+1
print(PowerSpectrum)
plt.plot(magk,PowerSpectrum)
plt.show()
##############################################
#Generate Bins for the reconstructed power spectrum
#############################################
binsize = 1
Bins = [int(i/binsize) for i in range(int(gridwidth*gridwidth/2))]
###################################################################
# generate all left of centre entries to signal frequency matrix and their conjugate symmetry entries
#######################################################
repeats = 0
while repeats < 1:
	noisevector = []
	datavector = []
	reconvector=[]
	truesignalvector=[]
	ftpowervector= []#ftpowervector
	n=1
	m=1
	while m < gridwidth/2: #create array for Re(s_k) + sqrt(-1) Im(s_k): <s_k s_kprime> = S(k) deltakkprime
		while n<gridwidth: #each imaginary and real part sampled from N(0,S(k))
			k = numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
			im=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*powerspectrumfunc(k), size =None)
			re=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*powerspectrumfunc(k), size =None)
			signalfreq[n][m]=re+1j*im
			signalfreq[-n][-m]=re-1j*im
			n=n+1
		n=1
		m=m+1
	#########################################
	# generate all centre column entries (except 0,0 which can be defined as a sample from N(0,0), which is a delta at 0)
	#############################################
	n=1
	m=int(gridwidth/2)
	while n<gridwidth/2:
		k = numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
		im=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*powerspectrumfunc(k), size = None)
		re=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*powerspectrumfunc(k), size = None)
		signalfreq[n][m]=re+1j*im
		signalfreq[-n][-m]=re-1j*im
		n=n+1
	########################################################
	# generate the edge entries as real samples from N(0,2*S(k))
	########################################################
	n=0
	m=0
	while n<gridwidth:
		re=numpy.random.normal(loc=0.0,scale =powerspectrumfunc(numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))), size = None)
		signalfreq[n][m]=re
		n=n+1
	m=1
	n=0
	
	while m<gridwidth:
		re=numpy.random.normal(loc=0.0,scale =
		powerspectrumfunc(numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+
		float(m-(gridwidth/2))*float(m-(gridwidth/2)))), size = None)
		signalfreq[n][m]=re
		m=m+1
	signalfreq[int(gridwidth/2)][int(gridwidth/2)] = powerspectrumfunc(0)
	#############
	#add noise
	#############
	signalpixel = numpy.fft.fft2(signalfreq) #generate ft into pixel space
	signalpixel2=signalpixel+0.0
	signalpixel2=phasefactor(signalpixel2)
	map2=[[0]*gridwidth for i in range (gridwidth)]
	n=0
	m=0
	while n<gridwidth: #form a data vector (plus some others)
		while m < gridwidth:
			noise=numpy.random.normal(loc=0.0, scale = sigmanoise, size = None)
			map[n][m]=signalpixel[n][m] + noise
			map2[n][m]=signalpixel2[n][m] + noise
			datavector.append(map[n][m])
			noisevector.append(noise)
			truesignalvector.append(signalpixel[n][m])
			m=m+1
		m=0
		n=n+1
	#######################
	#form power spectrum array
	#########################
	n=0
	m=0
	while m<gridwidth: #form a power spectrum map (plus some others)
		while n < gridwidth:
			k = numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
			powermap[n][m] = powerspectrumfunc(k)
			n=n+1
		n=0
		m=m+1
	############
	## Wiener
	#############
	ftpower = numpy.fft.fft2(powermap) #generate ft of powerspectrum map (Correlation Function)
	n=0
	m=0
	while m<gridwidth: #form a ftpower vector
		while n < gridwidth:
			ftpowervector.append(ftpower[n][m])
			n=n + 1
		n = 0
		m=m+1
	N = [0 for i in range(gridwidth*gridwidth)] #noise covariance
	filteredmap = [[0]*gridwidth for i in range(gridwidth)]
	##################
	## Forming N cov matrix
	#################################
	n=0
	while n < (gridwidth*gridwidth): #make diagonal noise cov matrix
		N[n] = sigmanoise*sigmanoise
		n=n+1
	######################
	## Forming S convariance
	#######################
	Sft = [0 for i in range(gridwidth*gridwidth)]
	#####################
	## Forming Sft convariance
	########################
	n = 0
	m = 0
	while m < gridwidth:
		while n < gridwidth:
			k = numpy.power((n-gridwidth/2)*(n-gridwidth/2) + (m-gridwidth/2)*(m-gridwidth/2),0.5)
			Sft[m*gridwidth + n] = powerspectrumfunc(k)
			n = n + 1
		m = m + 1
		n = 0

	reconvector = WienerMessengerField(N,Sft,datavector,30) #calculate the reconstructed map with 10 iterations
	n = 0
	m = 0
	while n < gridwidth:
		while m < gridwidth:
			filteredmap[n][m] = float(numpy.real(reconvector[n*gridwidth+m]))+1j*float(numpy.imag(reconvector[n*gridwidth+m]))
			m = m + 1
		m=0
		n = n+ 1
	#####################################
	#Reconstruct Power Spectrum
	#####################################
	FTReconmap = numpy.fft.fft2(filteredmap)/(gridwidth*gridwidth) # Fourier of Reconstructedmap
	n = 0
	m = 0
	counting = 0
	while n < gridwidth:
		while m < gridwidth:
			k =(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
			k=k-1 #shifts k forwards for easy maths
			k=int(k)
			k = Bins[k]
			ReconPowerSpectrum[k] = (ReconPowerSpectrum[k]*countk[k]+numpy.abs(FTReconmap[n][m]*pixelwidth))/(countk[k]+1)
			countk[k] = countk[k]+1
			m = m + 1
		n = n + 1
		m = 0
	repeats = repeats+1

kval= [0 for i in range(int(gridwidth*gridwidth*0.5))] #define same as magk
magkone =[0 for i in range(int(gridwidth*gridwidth*0.5))] # magnitude of the wave vector
i=0
while i < int(gridwidth*gridwidth*0.5): #create powerspectrum list for values along diagonal in k space
	magkone[i]=i
	i=i+1

ReconPowerSpectrum.insert(0, 0) #shifts k back so that the k=0 => S(k)=0
L=len(ReconPowerSpectrum)-1
ReconPowerSpectrum.pop(L)
countk.insert(0, 0)
L=len(countk)-1
countk.pop(L)

i=1
while i < len(countk): #delete empty values from
	if countk[i] == 0:
		countk.pop(i)
		ReconPowerSpectrum.pop(i)
		magkone.pop(i)
		i=0	
	i=i+1

newishlist=[numpy.sqrt(x) for x in magkone]
magkone=newishlist

i = int(gridwidth*numpy.sqrt(0.5))

while i < griddiag:
	PowerSpectrum.pop(int(gridwidth*numpy.sqrt(0.5)))
	magk.pop(int(gridwidth*numpy.sqrt(0.5)))
	i = i + 1

magk2, ReconPowerSpectrum2=PowerSpectrumGenerator(signalfreq)
x=numpy.log(-numpy.array(ReconPowerSpectrum)+numpy.array(ReconPowerSpectrum2))
y=numpy.log(numpy.array(ReconPowerSpectrum))
plt.subplot(1,3,1),plt.plot(magk,numpy.log(PowerSpectrum))
plt.subplot(1,3,2),plt.scatter(magkone,y)
plt.subplot(1,3,3),plt.scatter(magk2,x)
plt.show()
plt.savefig("ReconMessengerField")

###########################################
# plot the maps
###########################################
plt.subplot(2,2,1),plt.imshow(numpy.real(signalpixel2), cmap = "jet")
plt.ylabel("True Map")
plt.colorbar()
plt.subplot(2,2,2),plt.imshow(numpy.real(map2), cmap = "jet")
plt.ylabel("Map + Noise")
plt.colorbar()
plt.subplot(2,2,3),plt.imshow(numpy.real(phasefactor(filteredmap)), cmap = "jet")
plt.ylabel("Filtered Map")
plt.colorbar()
plt.subplot(2,2,4),plt.imshow(numpy.real(signalpixel2-(filteredmap)), cmap = "jet")
plt.ylabel("Difference")
plt.colorbar()
plt.show()

print ("done")