
# # Statistical Description of Data - Scientific Computing II - Alex Hernandez 

# ## Basic Statistical Tools

# Since these are just the basic stat tools I'm not goint to "implement" them on any real problems but just use `range(0,11)` to ensure the algorithms are working. Later after I've implemented most of the two from this chapter in _Numerical Recipes 3rd Edition_ I'll use them on some legitmate problems.
# 
# So lets begin... wait. Thats right &mdash; I'm using python for this library! Okay... now lets go!

# In[263]:

import matplotlib, math, sys, numpy


# >At best, you can substantiate a hypothesis by ruling out, statistically, a whole long list of competing hypotheses, every one that has ever been proposed. After a while your adversaries and competitors will give up trying to think of alternative hypotheses, or else will grow old and die, and _then your hypothesis will become accepted_. Sounds crazy, we know, but that's how science works!
# 
# >&ndash; from Numerical Recipes

# As I implement 

# In[119]:

def shortMean( data ):
    """ returns the mean:
        this function divide the sum of the data points by N-1
        instead of N, where N is the number of data points.
    """
    avg = 0.0
    l = float(len(data))
    for i in data:
        avg += i
    
    return avg/(l-1)


# In[47]:

shortMean(range(0,11))


# Out[47]:

#     5.5

# In[118]:

def variance( data ):
    """ return the variance """
    dataMean = shortMean(data)
    dataDiff = sum( map( lambda x: ( x - dataMean)**2, data ) )
    return dataDiff / ( float( len(data) - 1.0 ) )


# In[63]:

variance( range(0,11) )


# Out[63]:

#     11.275

# In[116]:

def deviation( data ):
    """ returns the standard deviation """
    return math.sqrt( variance(data) )


# In[69]:

deviation( range(0,11) )


# Out[69]:

#     3.357826678076163

# In[117]:

def stderror( data ):
    """ return the standard error """
    sigma = deviation( data )
    N = len(data)
    return sigma/(math.sqrt(N))


# In[73]:

stderror( range(0,11) )


# Out[73]:

#     1.0124228365658294

# Implement the _average deviation_ or the _mean absolute deviation_

# In[115]:

def avgdev( data ):
    """ return the average deviation:
        this value is helpful when variance = 0
    """
    m = shortMean( data )
    N = float(len(data))
    dataDiff = sum( map( lambda x: abs( x - m ), data ) )
    return dataDiff / N


# In[75]:

avgdev( range(0,11) )


# Out[75]:

#     2.772727272727273

# In[114]:

def skew( data ):
    """ returns the value of skew
        if skew > 0, the distribution drags from +inf
        if skew < 0, the distribution drags from -inf
    """
    m = shortMean( data )
    sigma = deviation( data )
    dataDiff = sum( map( lambda x: ( (x-m) / sigma )**3, data ) )
    return dataDiff / float(len(data))


# In[78]:

skew( range(0,11) )


# Out[78]:

#     -0.3995034715141182

# This makes sense for the `range(0,11)` as a negative value implies that the "tail" of the distribution drags out toward the negative x and since the distribution is just the $y=x$ function, this makes sense.
# 
# Now I implement the kurtosis function

# In[112]:

def kurtosis( data ):
    """ Return the kurtosis, k, of the data
        if k > 0, the peak of the data is sharp, like a mountain ( leptokurtic )
        if k < 0, the peak is soft, like a loaf of bread ( platykurtic )
        if k = 0, the peak is somewhere in between the two ( mesokuritc )
    """
    m = shortMean( data )
    sigma = deviation( data )
    dataDiff = sum( map( lambda x: ( (x-m) / sigma )**4, data ) )
    return (dataDiff / float(len(data))) - 3


# In[113]:

kurtosis( range(0,11) )


# Out[113]:

#     -1.4813250672317244

# The original `varaince` function was expanded using the belief that limiting distribution was gaussian. I would now implement the variance in terms of the binomial distribution but the direct implementation suffers from a magnification of the roundoff error and is slow computationaly. The way fix this is called the _correct two-pass algorithm_ which I implement below

# In[83]:

def ctpVar( data ):
    N = float(len(data))
    m = shortMean( data )
    diff2 = sum( map( lambda x: (x - m)**2, data) )
    diff = (1.0/N)*sum( map( lambda x: (x - m), data) )
    return ((1.0/(N-1.0))*(diff2 - diff**2 ))
    


# In[84]:

ctpVar( range(0,11) )


# Out[84]:

#     11.25

# now re-implement standard deviation in terms of `cptVar`

# In[85]:

def stdev( data ):
    return math.sqrt( cptVar( data ) )


# So far, I have been implment these algorithms straight from the equations. Now I will implement a more efficient algorithm that incorporates all of these together. This will reduce the computation needed because somethings like the mean get repeatedly calculated if all these algrothims were brought together as is.

# In[109]:

def moment( data ):
    """ Return the following statistical values:
        average, average deviation, standard deviation, varaince, skew, and kurtosis.
    """
    n = float(len(data))
    ep = s = p = 0.0
    if( n <= 1 ):
        raise Exception("n must be at least 2 in moment")
    s = sum( data )
    ave = s/n
    adev=var=skew=curt=0.0
    for d_i in data:
        s=d_i-ave
        adev += abs( s )
        ep += s
        p=s*s
        var += p
        p *= s
        skew += p
        p *= s
        curt += p
    adev /= n
    var=(var-ep*ep/n)/(n-1.0)
    sdev=math.sqrt(var)
    if( var != 0.0 ):
        skew /= (n*var*sdev)
        curt=curt/(n*var*var)-3.0
    else:
        skew = float('nan')
        curt = float('nan')
    
    return [ave, adev, sdev, var, skew, curt]
    


# In[110]:

moment( range(0,11) )


# Out[110]:

#     [5.0, 2.727272727272727, 3.3166247903554, 11.0, 0.0, -1.5289256198347108]

# ## Special Functions Code
# 
# Before I can move onto Student's t-test, F-test, Chi-square and the like I need to go to Sections 6.1-4,14 and implment the algorithms in these sections. The book _Numerical Recipes_ uses specialized code form these sections that are prolly unique to the book.
# 
# The first of these is the natural logarithm of the gamma function, $ \ln{ \left[ \Gamma(x) \right] } $

# In[153]:

def gammln( xx ):
    """ returns the natuaral logarithm of the gamma function at xx
        Uses: the factorial of xx-1 can be computed by doing
        math.exp( gammln( xx ) ) about eqaul to math.factorial(xx-1)
        ( This is a more efficeint computation. )
    """
    cof = [ 57.1562356658629235, -59.5979603554754912,
            14.1360979747417471, -0.491913816097620199, .339946499848118887e-4,
            .465236289270485756e-4, -.983744753048795646e-4, .158088703224912494e-3,
            -.210264441724104883e-3, .217439618115212643e-3, -.164318106536763890e-3,
            .844182239838527433e-4, -.261908384015814087e-4, .368991826595316234e-5]
    if( xx <= 0 ):
        raise Exception("bad arg in gammln")
    y=x=xx
    tmp = x+5.24218750000000000
    tmp = (x+0.5)*math.log(tmp)-tmp
    ser = 0.999999999999997092
    for j in cof:
        y+=1
        ser += j/y
    return tmp+math.log(2.5066282746310005*ser/x)


# Test this using $3! = \ln{ \left[ \Gamma( 3 + 1) \right] } = 6 $

# In[154]:

# just check to make sure it worksmath.exp(gammln( 4 ))
print "3!= {0} = e^(ln[gamma(4)]) = {1}".format(math.exp(gammln(4)),math.factorial(3))


# Out[154]:

#     3!= 6.0 = e^(ln[gamma(4)]) = 6
# 

# success!
# 
# now implement the function that returns the first 170 factorial numbers

# In[426]:

def factrl(x):
    """ return the factorial of x and x is between 0 and 171.
    On first call, the function creates a list of the numbers and further calls
    just reference this list.
    """
    if not factrl.init:
        factrl.a[0] = 1.0
        for i in range(1,171):
            factrl.a[i] = i*factrl.a[i-1]
        factrl.init = True
    return factrl.a[x] 
factrl.a = [i*1.0 for i in range(0,171)]
factrl.init = False


# In[428]:

print factrl(5)
print factrl(26)


# Out[428]:

#     120.0
#     4.03291461127e+26
# 

# The problem with this function, though it is useful is that numbers $23!$ and above are only represented approximately. It's better to create a funciton that, when intialized, stores the natural logs of the factorials of first 2000 integers. This allows exact representations at the cost of memory storage. 

# In[236]:

def factln(x):
    """ returns the natural logarithms of the factorial of x and x is between 0 and 2000.
    On first call, the function creates a list of the numbers and further calls
    just reference this list.
    """
    if not factln.init:
        for i in xrange(0,factln.NTOP): factln.a[i] = gammln(i+1.0)
        factln.init = True
    if(x<0): raise Exception("negative arge in factln")
    if(x<factln.NTOP): return factln.a[x]
factln.init = False 
factln.NTOP = 2000
factln.a = [i*1.0 for i in range(0,factln.NTOP)]


# Run `factln` for a range of different values.

# In[229]:

[factln(i) for i in range(0,11)]


# Out[229]:

#     [0.0,
#      0.0,
#      0.6931471805599458,
#      1.7917594692280538,
#      3.1780538303479458,
#      4.787491742782044,
#      6.579251212010101,
#      8.525161361065415,
#      10.604602902745253,
#      12.801827480081469,
#      15.10441257307552]

# Explicitly solve the for factarial 3

# In[230]:

print math.exp(factln(4))


# Out[230]:

#     24.0
# 

# In[ ]:




# now using both object of `factrl` and `factln`, I implment a function which returns the binomial coeffiecent for (n k).

# In[244]:

def bico(n,k):
    """ Returns the binomial coefficient (n k) as a floating point number. """
    if( n<0 or k<0 or k>n ): raise Exception("bad args in bico")
    if(n<171): return math.floor(0.5+factrl(n)/(factrl(k)*factrl(n-k)))
    else: return math.floor(0.5+math.exp(factln(n)-factln(k)-factln(n-k)))


# In[246]:

bico(4, 2)


# Out[246]:

#     6.0

# Now, implementing the `beta` function, $B(z,w) = \frac{ \Gamma\left( z \right) \Gamma\left(w \right)}{\Gamma \left( z + w \right) } $.

# In[247]:

def beta(z,w):
    """Returns the value of the beta function B(z,w)"""
    return math.exp(gammln(z)+gammln(w)-gammlb(z+w))


# In[261]:

class Gauleg18(object):
    ngau = 18
    y = [ 0.0021695375159141994, 
          0.011413521097787704,0.027972308950302116, 0.051727015600492421,
          0.082502225484340941, 0.12007019910960293, 0.16415283300752470,
          0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
          0.39843234186401943, 0.46921971407375483, 0.54413605556657973,
          0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
          0.87126389619061517, 0.95698180152629142]
    w = [ 0.0055657196642445571, 
          0.012915947284065419, 0.020181515297735382, 0.027298621498568734,
          0.034213810770299537, 0.040875750923643261, 0.047235083490265582,
          0.053244713977759692, 0.058860144245324798, 0.064039797355015485,
          0.068745323835736408, 0.072941885005653087, 0.076598410645870640,
          0.079687828912071670, 0.082187266704339706, 0.084078218979661945,
          0.085346685739338721, 0.085983275670394821]


# The following is a an object which houses all the functions need for an incomplete integral on Gamma. The two main functions are `gammp` and `gammq`. `gammp` returns the value of the integral 
# > $$ P(a,x) \equiv \frac{ \gamma(a,x) }{ \Gamma(a) } \equiv \frac{1}{ \Gamma(a) } \int_{0}^{x} e^{-t} t^{a-1} dt \\
# \text{where }(a > 0 ) \\
# \text{and } P(a,0) = 0, \space\space\space P(a,\infty) = 1$$

# In[303]:

class Gamma(Gauleg18):
    """Object for incomplete gamma function. 
    Gauleg18 provides coefficients for Gauss-Legendre quadrature
    """
    ASWITCH=100
    EPS = sys.float_info[8] # get the max value of a float
    FPMIN = sys.float_info[3]/EPS
    
    def gammp(self,a,x):
        """ Returns the incomplete gamma function P(a,x)."""
        if(x<0.0 or a <= 0.0): raise Exception("bad args in gammp")
        if(x == 0.0): return 0.0
        elif( int(a) >= self.ASWITCH): return self.gammpapprox(a,x,1)
        elif( x < a+1.0): return self.gser(a,x)
        else: return 1.0-self.gcf(a,x)
    
    def gammq(self,a,x):
        """Returns the incomplete gamma funciton Q(a,x)=1-P(a,x)."""
        if(x<0.0 or a<=0.0): throw("bad args in gammq")
        if(x==0.0): return 1.0
        elif(int(a)>=self.ASWITCH): return self.gammpapprox(a,x,0)
        elif(x<a+1.0): return 1.0-self.gser(a,x)
        else: return self.gcf(a,x)
        
    def gser(self,a,x):
        """Returns the incomplete gamma function P(x,a) evaluated by its series representation.
        Also set ln(Gamma(a)) as gln. User should not call directly.
        """
        self.gln=gammln(a)
        ap=a
        dl=sm=1.0/a
        while(True):
            ap += 1
            dl *= x/ap
            sm += dl
            if( abs(dl) < abs(sm)*self.EPS):
                return sm*math.exp(-x+a*math.log(x)-self.gln)
        
    def gcf(self,a,x):
        """Return the incomplete gamm function Q(a,x) evaluated by its continued fraction representation.
        Alse sets ln(gamma(a)) as gln. User should not call directly.
        """
        self.gln=gammln(a)
        b=x+1.0-a
        c=1.0/self.FPMIN
        d=1.0/b
        h=d
        i=1
        while( True ):
            an = -i*(i-a)
            b += 2.0
            d=an*d+b
            if( abs(d) < self.FPMIN): d=FPMIN
            c=b+an/c
            if( abs(c) < self.FPMIN): c=FPMIN
            d=1.0/d
            dl=d*c
            h *= dl
            if( abs(dl-1.0) <= self.EPS): break
        return math.exp(-x+a*math.log(x)-self.gln)*h
    
    def gammpapprox(self,a,x,psig):
        """Incomplete gamma by quadrature. Returns P(a,x) or Q(a,x), when psig is 1 or 0, respctively.
        User should not call directly.
        """
        xu=t=0.0
        a1 = a-1.0
        lna1 = math.log(a1)
        sqrta1 = math.sqrt(a1)
        gln = gammln(a)
        # see how far to integrate into the tail
        if(x>a1):
            xu=max(a1+11.5*sqrta1, x+6.0*sqrta1)
        else:
            xu=max(0.0,min(a1-7.5*sqrta1,x-5.0*sqrta1))
        sm = 0.0
        for j in xrange(0,self.ngau):
            t = x+(xu-x)*y[j]
            sm += w[j]*math.exp(-(t-a1)+a1*(math.log(t)-lna1))
        ans = sm*(xu-x)*math.exp(a1*(lna1-1.0)-self.gln)
        if( psig == 1 ):
            if( ans > 0.0 ):
                return 1.0-ans
            else:
                return ans
        else:
            if( ans >= 0.0 ):
                return 0.0
            else:
                return 1.0+ans
        
        def invgammp(self, p, a):
            """Return x such that P(a,x)=p for an argument p between 0 and 1
            Inverse function on x of P(a,x).
            """
            pp=0.0
            EPS=1.0e-8
            a1=a-1
            gln=gammln(a)
            if(a<=0.0): raise Exception("a must be pos in invgammp")
            if(p>=1.0): return max(100.0,a+100.0+math.sqrt(a))
            if(p<=0.0): return 0.0
            if(a>1.0):
                lna1=math.log(a1)
                afac = math.exp(a1*(lna1-1.0)-self.gln)
                if(p<0.5): pp=p
                else: pp = 1.0-p
                t = math.sqrt(-2.0*log(pp))
                x=(2.30753+t*0.27061)/(1.0+t*(0.99229+t*0.04481))-t
                if(p<0.5): x = -x
                x = max(1.0e-3, a*(1.0-(1.0/(9.0*a))-(x/(3.0*math.sqrt(a))))**3)
            else:
                t = 1.0 - a*(0.253+a*0.12)
                if(p<t): x = (p/t)**(1.0/a)
                else: x = 1.0-math.log(1.0-(p-t)/(1.0-t))
            for j in xrange(0,12):
                if(x<=0.0): return 0.0
                err = gammp(a,x) - p
                if(a>1.0): t = afac*math.exp(-(x-a1)+a1*(log(x)-lna1))
                else: t = math.exp(-x+a1*math.log(x)-gln)
                u = err/t
                t = u/(1.0-0.5*min(1.0,u*(((a-1.0)/x)-1.0)))
                if(x<=0.0): x = 0.5*(x+t)
                if( abs(t) < self.EPS*x): break
            return x


# In[306]:

gamm = Gamma()
print gamm.gammp(0.1,2.0)
print gamm.gammq(0.1,2.0)
print gamm.gammp(0.1,2.0)+gamm.gammq(0.1,2.0)


# Out[306]:

#     0.994376487882
#     0.00562351211827
#     1.0
# 

# Well, it looks like it works. I'm going to have to sit down and really think about the all the different test cases.

# In[352]:

class Erf(object):
    """Object for error function and related funcitons"""
    ncof=28
    cof = [ -1.3026537197817094, 6.4196979235649026e-1, 
           1.9476473204185836e-2, -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
            3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5, 
            -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
            6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
            9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13,
            -1.12708e-13, 3.81e-16, 7.106e-15, -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17]
    
    def erf(self, x):
        """Retrun erf(x) for any x"""
        if(x>=0.0): return 1.0-self.erfccheb(x)
        else: return self.erfccheb(-x)-1.0
        
    def erfc(self,x):
        """Return erfc(x) for any x"""
        if(x>=0.0): return self.erfccheb(x)
        else: return 2.0-self.erfccheb(-x)
        
    def erfccheb(self,z):
        """Evaluate the equation: erfc(z) about = t exp[-z**2+P(t)], z>0
        using stored Chebyshev cooeficients. User should not call directly."""
        d=dd=0.0
        if(z<0.0): raise Exception("erfcched requires nonnegative argument")
        t = 2.0/(2.0+z)
        ty = 4.0*t - 2.0
        for j in range(self.ncof-1,0,-1):
            tmp = d
            d = ty*d - dd + self.cof[j]
            dd = tmp
        return t*math.exp(-z*z + 0.5*(self.cof[0]+ty*d)-dd)
    
    def inverfc(self,p):
        """Iverse of complmentary error function. Return x such that erfc(x)=p for argument p
        between 0 and 2"""
        err = 0.0
        if(p>=2.0): return -100.0
        if(p<=0.0): return 100.0
        if(p<1.0): pp = p
        else: pp = 2.0 - p
        t = math.sqrt(-2.0*log(pp/2.0))
        x = -0.70711*((2.30753+t*0.27061)/(1.0+t*(0.99229+t*0.04481))-t)
        for j in xrange(0,2):
            err = erfc(x) - pp
            x += err/(1.12837916709551257*math.exp(-(x**2))-x*err)
        if(p<1.0): return x
        else: return -x
        
    def inverf(self,p):
        """Inverse of the error funciton. Return x such that erf(x) = p for arguement between -1
        and 1"""
        return inverfc(1.0-p)


# In[353]:

my_err = Erf()


# In[354]:

print my_err.erf(0.1)
print my_err.erf(0.5)
print my_err.erf(1.0)
print my_err.erf(5.0)


# Out[354]:

#     0.112754082424
#     0.520357177501
#     0.842681743299
#     0.999999999998
# 

# In[356]:

print my_err.erf(-0.1)
print my_err.erf(-0.5)
print my_err.erf(-1.0)
print my_err.erf(-5.0)


# Out[356]:

#     -0.112754082424
#     -0.520357177501
#     -0.842681743299
#     -0.999999999998
# 

# In[362]:

def erfcc(x):
    """Returns the complementary error function erfc(x) with fractional error everythwere
    less than 1.2x10**(-7)"""
    t=ans=z=abs(x)
    t=2.0/(2.0+z)
    ans=t*math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+
        t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*
        (1.48851587+t*(-0.82215223+t*0.17087277)))))))))
    if(x>=0.0): return ans
    else: return 2.0-ans


# In[360]:

erfcc(0.1)


# Out[360]:

#     0.887537071229425

# ### Exponential Integrals

# In[414]:

def expint(n,x):
    """Evaluates the exponential integral E_n(x). """
    MAXIT=100
    EULER=0.577215664901533
    EPS=sys.float_info[8]
    BIG=sys.float_info[1]*EPS
    nm1=n-1
    if(n<0 or x<0.0 or (x==0.0 and (n==0 and n==1))):
       raise Exception("bad arguments in expint")
    if(n==0): ans=math.exp(-x)/x
    else:
        if(x>1.0):
            b=x+n
            c=BIG
            d=1.0/b
            h=d
            for i in range(1,MAXIT):
                a = -i*(nm1+i)
                b += 2.0
                d=1.0/(a*d+b)
                c=b+a/c
                dl=c*d
                h *= dl
                if( abs(dl-1.0) <= EPS ):
                    ans=h*math.exp(-x)
                    return ans
            raise Excpetion("continued fraction failed in expint")


# In[415]:

def ei(x):
    """Computes the exponential integral Ei(x) for x > 0"""
    MAXIT=100
    EULER=0.577215664901533
    EPS=sys.float_info[8]
    FPMIN=sys.float_info[3]/EPS
    k = 0
    if(x<=0.0): raise Exception("Bad argument in ei")
    if(x<FPMIN): return log(x)+EULER
    if(x<= -math.log(EPS)):
        sm = 0.0
        fact = 1.0
        for k in range(1,MAXIT+1):
            fact *= x/k
            term = fact/k
            sm += term
            if(term<EPS*sm): break
        if(k>MAXIT): raise Exception("Series failed in ei")
        return sm+math.log(x)+EULER
    else:
        sm = 0.0
        term=1.0
        for k in range(1,MAXIT+1):
            prev=term
            term *= k/x
            if(term<EPS): break
            if(term<prev): 
                sm+=term
            else:
                sm -= prev
                break
        return math.exp(x)*(1.0+sm)/x
    


# In[416]:

class Beta(Gauleg18):
    """ Obhect for incomplete beta function. Gauleg 18 provide coefficients for Gauss-Lgendre
    qaudrature.
    """
    SWITCH=3000
    def __init__(self):
        self.EPS = sys.float_info[8]
        self.FPMIN = sys.float_info[3]/self.EPS
        
    def betai(self, a,b,x):
        """ Return incomplete beta funciton I_x(a,b) for a positive a and b, and x between 0 and 1.
        """
        if(a<=0.0 or b<=0.0): raise Exception("Bad a or b in routine betai")
        if(x<0.0 or x > 1.0): raise Exception("Bad x in rountine betai")
        if(x == 0.0 or x == 1.0): return x
        if(a > self.SWITCH and b > self.SWITCH): return betaiapprox(a,b,x)
        bt=math.exp(gammln(a+b)-gammln(a)-gammln(b)+a*math.log(x)+b*math.log(1.0-x))
        if(x<(a+1.0)/(a+b+2.0)): return bt*self.betacf(a,b,x)/a
        else: return 1.0-bt*self.betacf(b,a,1.0-x)/b
        
    def betacf(self,a,b,x):
        """Evaluates continued fraction for incomplete beta funciton by modified Lentz's
        method. User should not call directly.
        """
        qab=a+b
        qap=a+1.0
        qam=a-1.0
        c=1.0
        d=1.0-qab*x/qap
        if( abs(d) < self.FPMIN): d=self.FPMIN
        d=1.0/d
        h=d
        for m in range(1,10000):
            m2=2*m
            aa=m*(b-m)*x/((qam+m2)*(a+m2))
            d=1.0+aa*d
            if(abs(d)<self.FPMIN): d=self.FPMIN
            c=1.0+aa/c
            if(abs(c)<self.FPMIN): c=self.FPMIN
            d=1.0/d
            h *= d*c
            aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d=1.0+aa*d
            if(abs(d)<self.FPMIN): d=self.FPMIN
            c=1.0+aa/c
            if(abs(c)<self.FPMIN): c=self.FPMIN
            d=1.0/d
            dl = d*c
            h *= dl
            if(abs(dl-1.0)<=self.EPS): break
        return h
    
    def betaiapprox(self,a,b,x):
        """ Incomplete beta by quadrature. Reutrns I_x(a,b). User should not call directly.
        """
        a1=a-1.0
        b1=b-1.0
        mu=a/(a+b)
        lnmu=math.log(mu)
        lnmuc=math.log(1.0-mu)
        t = math.sqrt(a*b/( ((a+b)**2)*(a+b+1.0)))
        if(x> a/(a+b)):
            if(x>=1.0): return 1.0
            xu = min(1.0, max(mu+10.0*t, x+5.0*t))
        else:
            if(x<=0.0): return 0.0
            xu = max(0.0, min(mu-10.0*t, x-5.0*t))
        sm = 0.0
        for j in range(0,18):
            t = x + (xu-x)*self.y[j]
            sm += w[j]*math.exp(a1*(log(t)-lnmu)+b1*(log(1-t)-lnmuc))
        ans = sum*(xu-x)*math.exp(a1*lnmu-gammln(a)+b1*lnmuc-gammln(b)+gammln(a+b))
        if(ans>0.0): return 1.0-ans
        else: return -ans
        
    def invbetai(self,p,a,b):
        """Inverse of incomplete beta funciton. Returns x such that I_x(a,b) = p 
        for argument p between 0 and 1.
        """
        pp=t=u=err=x=al=h=w=afac=0.0
        j = 0
        EPS = 1.0e-8
        a1=a-1.0
        b1=b-1.0
        if(p <= 0.0): return 0.0
        elif(p >= 1.0): return 1.0
        elif(a >= 1.0 and b >= 1.0):
            if(p<0.5): pp = p
            else: pp = 1.0 - p
            t = math.sqrt(-2.0*math.log(pp))
            x = (2.30753+t+0.27061)/(1.0+t*(0.99229+t*0.04481))-t
            if(p<0.5): x = -x
            al = ( x**2 -3.0)/6.0
            h = 2.0/(1.0/(2.0*a-1.0)+1.0/(2.0*b-1.0))
            w = (x*math.sqrt(al+h)/h)-(1.0/(2.0*b-1)-1.0/(2.0*a-1.0))*(al+5.0/6.0-2.0/(3.0*h))
            x = a/(a+b*math.exp(2.0*w))
        else:
            lna = math.log(a/(a+b))
            lnb = math.log(b/(a+b))
            t = math.exp(a*lna)/a
            u = math.exp(b*lnb)/b
            w = t + u
            if(p<(t/w)): x = (a*w*p)**(1.0/a)
            else: x = 1.0 - (b*w*(1.0-p))**(1.0/b)
        afac = -gammln(a)-gammln(b)+gammln(a+b)
        for j in xrange(0,10):
            if(x ==0.0 or x == 1.0): return x
            err = self.betai(a,b,x)-p
            t = math.exp(a1*math.log(x)+b1*math.log(1.0-x)+afac)
            u = err/t
            t = u/(1.0-0.5*min(1.0, u*(a1/x - b1/(1.0-x))))
            x -= t
            if(x<=0.0): x = 0.5*(x+t)
            if(x>=1.0): x = 0.5*(x+t+1.0)
            if( abs(t) < EPS*x and j>0): break
        return x


# ## Satisitical Functions Objects
# 
# The next section of code is set objects which describe common probability distributions and will be used to implment tests on the statistical funcitons.
# 
# >In this section we survey a number of... common distributions in a unified way, giving, in each case, routines for computing the probability density funciton $p(x)$; the _cumulative density function_ of _cdf_, writing $P(<x)$; and the inverse of hte cumalative density function $x(P)$. The latter funciton is needed fo finding the values of $x$ associated with specficed _percentile points_ or _quantiles_ in significane test, for exampl, the _0.5%_, _5%_, _95%_ or _99.5%_ points.
# 
# 
# ### The Normal Distribution
# 
# >$ x \sim N(\mu , \sigma ), \space\space\space \sigma \gt 0 $
# 
# The probability density funciton:
# >$ p(x) = \frac{1}{ \sqrt{2 \pi} \sigma} \exp{ \left( - \frac{1}{2} \left[ \frac{x-\mu}{\sigma} \right]^2 \right) } $
# 
# The cumulative distribution funciton, the probaility of a value $\ge x$,
# >$ \text{cdf} \equiv P( \ge x) \equiv \int_{-\infty}^{x} p(x')dx' = \frac{1}{2}\text{erfc}\left( - \frac{1}{2} \left[ \frac{x-\mu}{\sigma} \right] \right) $
# 
# The _inverse cdf_ can be calculated in terms of the inverse of $\text{erfc}$
# >$ x(P) = \mu - \sqrt{2}\sigma \text{erfc}^{-1}(2P) $

# In[418]:

class Normaldist(Erf):
    """Normal distribution, derived from the rror funciton Erf
    """
    mu = 0.0
    sig = 0.0
    
    def __init__(self,mmu=0.0,ssig=1.0):
        """Constructor. Initialize with mu and sig. The deafulat with no arguments is N(0,1).
        """
        self.mu = mmu
        self.sig = ssig
        if(self.sig<=0.0): raise Exception("bad sig in Normaldist")
    
    def p(self,x):
        """Return probability density function.
        """
        return (0.398942280401432678/self.sig)*math.exp(-0.5*((x-self.mu)/self.sig)**2)
    
    def cdf(self,x):
        """Return cumulative distribution funciton.
        """
        return 0.5*self.erfc(-0.707106781186547524*(x-self.mu)/self.sig)
    
    def invcdf(self,p):
        """Return inverse cumulative distribution funciton.
        """
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Normaldist")
        return -1.41421356237309505*self.sig*self.inverfc(2.0*p)+self.mu


# In[ ]:




# ### Caushing Dstribution
# 

# In[424]:

class Cauchydist(object):
    """Caushy distribution
    """
    mu = 0.0
    sig = 0.0   
    def __init__(self,mmu=0.0,ssig=1.0):
        """Constructor. Initialize with mu and sigma. The default with no arguments is Cauchy(0,1).
        """
        self.mu=mmu
        self.sig=ssig
        if(self.sig<=0.0): raise Exception("bad sig in Cauchydist")
            
    def p(self,x):
        """Return probability density function.
        """
        return 0.318309886183790671/(self.sig*(1.0+((x-self.mu)/self.sig)**2))
    
    def cdf(self,x):
        """Return cumulative distribution funciton.
        """
        return (0.5+0.318309886183790671*math.atan2(x-self.mu,self.sig))
    
    def invcdf(self,p):
        """Return inverse cumulative distribution function.
        """
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Cauchydist")
        return self.mu+self.sig*math.tan(3.14159265358979324*(p-0.5))


# In[423]:

math.atan2(1.2,1.0)


# Out[423]:

#     0.8760580505981934

# ### Student-t Distribution

# In[396]:

class Studenttdist(Beta):
    nu=0.0
    mu=0.0
    sig=0.0
    np=0.0
    fac=0.0
    
    def __init__(self,nnu,mmu=0.0,ssig=1.0):
        """Constructor. Initialize with nu, mu, and sigma. The default with on argument
        is Strudent(nu,0,1)
        """
        self.nu=nnu
        self.mu=mmu
        self.sig=ssig
        if(self.sig<=0.0 and self.nu<=0.0): raise Exception("bad sig,nu in Studenttdist")
        self.np = 0.5*(self.nu+1.0)
        self.fac=gammln(self.np)-gammln(0.5*self.nu)
        
    def p(self,t):
        """Return probability density function."""
        return math.exp(-self.np*math.log(1.0+(((t-self.mu)/self.sig)**2)/self.nu)+self.fac)
    
    def cdf(self,t):
        """Return cumulative distribution function."""
        p = 0.5*self.betai(0.5*self.nu, 0.5, self.nu/(self.nu+(((t-self.mu)/self.sig)**2)))
        if(t >= self.mu): return 1.0 - p
        else: return p
        
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Studenttdist")
        x = self.invbetai(2.0*min(p,1.0-p), 0.5*self.nu, 0.5)
        x = self.sig*math.sqrt(self.nu*(1.0-x)/x)
        if(p>=0.5): return mu+x
        else: return mu-x
        
    def aa(self,t):
        """Return the two-tailed cdf A(t|nu)."""
        if(t<0.0): raise Exception("bad t in Studenttdist")
        return 1.0-self.betai(0.5*self.nu,0.5, self.nu/(self.nu+(t**2)))
    
    def invaa(self,p):
        """Return the inverse, namely t such that p = A(t|nu)."""
        if(p<0.0 or p>=1.0): raise Exception("bad p in Studenttdist")
        x = invbetai(1.0-p, 0.5*self.nu, 0.5)
        return math.sqrt(self.nu*(1.0-x)/x)


# In[ ]:




# ### Logistic Distribution

# In[397]:

class Logisticdist(object):
    """Logistic distribution."""
    mu=0.0
    sig=0.0
    
    def __init__(self,mmu=0.0,ssig=1.0):
        """Construtor. Initialize with mu and simga. The default with no arguments is Logistic(0,1)."""
        self.mmu=0.0
        self.ssig=1.0
        if(self.sig<=0.0): raise Exception("bad sig in Logisticdist")
    
    def p(self,x):
        """Return probability density function."""
        e = math.exp(-abs(1.81379936423421785*(x-self.mu)/self.sig))
        return 1.81379936423421785*e/(self.sig*((1.0+e)**2))
    
    def cdf(self,x):
        """Return cumulative distribution function."""
        e = math.exp(-abs(1.81379936423421785*(x-self.mu)/self.sig))
        if(x>=self.mu): return 1.0/(1.0+e)
        else: return e/(1.0+e)
        
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Logisticdist")
        return self.mu + 0.551328895421792049*self.sig*math.log(1.0-p)
    


# In[ ]:




# ### Exponential Distribution

# In[398]:

class Expondist(object):
    """Exponential Distribution"""
    bet = 0.0
    
    def __init__(self,bbet):
        """Constructor. Initialize with beta."""
        self.bet = bbet
        if(self.bet<=0.0): raise Exception("bad bet in Expondist")
    
    def p(self,x):
        """Return probability density function."""
        if(x<0.0): raise Exception("bad x in Expondist")
        return self.bet*math.exp(-self.bet*x)
    
    def cdf(self,x):
        """Return cumulative distribution function."""
        if(x<0.0): raise Exception("bad x in Expondist")
        return 1.0-math.exp(-self.bet*x)
    
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<0.0 or p>=1.0): raise Exception("bad p in Expondist")
        return -math.log(1.0-p)/self.bet


# In[ ]:




# ### Lognormal Distribution

# In[399]:

class Lognormaldist(Erf):
    """Lognormal distribution, derived from the error function Erf."""
    mu=0.0
    sig=0.0
    def __init__(self,mmu=0.0,ssig=1.0):
        self.mu = mmu
        self.sig = ssig
        if(self.sig<=0.0): raise Exception("bad sig in Lognormaldist")
        
    def p(self,x):
        """Return probability density function."""
        if(x<0.0): raise Exception("bad x in Lognormaldist")
        if(x == 0.0): return 0.0
        return (0.398942280401432678/(self.sig*x))*math.exp(-0.5*((math.log(x)-self.mu)/self.sig)**2)
    
    def cdf(self,x):
        """Return cumulative distribution funciton."""
        if(x<0.0): raise Exception("bad x in Lognormaldist")
        if(x==0.0): return 0.0
        return 0.5*self.erfc(-0.707106781186547524*(math.log(x)-self.mu)/self.sig)
    
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Lognormaldist")
        return math.exp(-1.4142135623730505*self.sig*self.inverfc(2.0*p)+self.mu)


# In[ ]:




# ### Chi-Sqaure Distribution

# In[400]:

class Chisqdist(Gamma):
    """Chi^2 distribution, derived from the gamma funciton Gamma."""
    nu=0.0
    fac=0.0
    
    def __init__(self,nnu):
        """Constructor. Initialize with nu."""
        self.nu=nnu
        if(self.nu<=0.0): raise Exception("bad nu in Chisqdist")
        self.fac = 0.693147180559945309*(0.5*self.nu)+gammln(0.5*self.nu)
        
    def p(self,x2):
        """Return probability density function."""
        if(x2<=0.0): raise Exception("bad x2 in Chisqdist")
        return math.exp(-0.5*(x2-(self.nu-2.0)*math.log(x2))-self.fac)
    
    def cdf(self,x2):
        """Return cumulative distribution function."""
        if(x2<0.0): raise Exception("bad x2 in Chisqdist")
        return gammp(0.5*self.nu,0.5*x2)
    
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<0.0 or p>=1.0): raise Exception("bad p in Chisqdist")
        return 2.0*self.invgammp(p,0.5*self.nu)


# In[ ]:




# ### Gamma Distribution

# In[401]:

class Gammadist(Gamma):
    """Gamma distribution, derived from the gamma function Gamma."""
    alph=0.0
    bet=0.0
    fac=0.0
    
    def __init__(self,aalph,bbet=1.0):
        """Constructor. Initialize with alpha and beta"""
        self.alph=aalph
        self.bet=bbet
        if(self.alph<=0.0 or self.bet<=0.0): 
            raise Exception("bad alph,bet in Gammadist")
        self.fac = self.alph*math.log(self.bet)-gammln(self.alph)
        
    def p(self,x):
        """Return probability density function."""
        if(x<=0.0): raise Exception("bad x in Gammadist")
        return math.exp(-self.bet*x+(self.alph-1.0)*math.log(x)+self.fac)
    
    def cdf(self,x):
        """Return cumulative distribution function."""
        if(x<0.0): raise Exception("bad x in Gammadist")
        return self.gammp(self.alph,self.bet*x)
    
    def invcdf(self,x):
        """Return inverse cumulative distribution funciton."""
        if(p<0.0 or p >= 1.0): raise Exception("bad p in Gammadist")
        return self.invgammp(p,self.alph)/self.bet


# In[401]:




# ### F-Distribution

# In[405]:

class Fdist(Beta):
    """F distribution derived from the beta function Beta"""
    nu1=0.0
    nu2=0.0
    fac=0.0
    
    def __init__(self,nnu1, nnu2):
        """Constructor. Initialize with nu1 and nu2"""
        self.nu1=nnu1
        self.nu2=nnu2
        if(self.nu1<=0.0 or self.nu2<= 0.0): 
            raise Exception("bad nu1, nu2 in Fdist")
        self.fac = 0.5*(self.nu1*math.log(self.nu1)+self.nu2*math.log(self.nu2))+gammln(0.5*(self.nu1+self.nu2))-gammln(0.5*self.nu1)-gammln(0.5*self.nu2)
    
    def p(self,f):
        """Return probability density function."""
        if(f<=0.0): raise Exception("bad f in Fdist")
        return math.exp((0.5*nu1-1.0)*math.log(f)-0.5(self.nu1+self.nu2)*
                        math.log(self.nu2+self.nu1*f)+self.fac)
    
    def cdf(self,f):
        """Return cumulative distribution function"""
        if(f<0.0): raise Exception("bad f in Fdist")
        return self.betai(0.5*self.nu1,0.5*self.nu2,
                          self.nu1*f/(self.nu2+self.nu1*f))
    
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Fdist")
        x = self.inbetai(p,0.5*self.nu1,0.5*self.nu2)
        return self.nu2*x/(self.nu1*(1.0-x))


# In[ ]:




# ### Beta Distribution

# In[407]:

class Betadist(Beta):
    """Beta distribution, derived from the beta function Beta."""
    alph=0.0
    bet=0.0
    fac=0.0
    
    def __init__(self, aalph, bbet):
        """Constructor. Initialize with alpha and beta"""
        self.alph=aalph
        self.bet=bbet
        if(self.alph<=0.0 or self.bet<=0.0): 
            raise Exception("bad alph, bet in Betadist")
        self.fac = gammln(self.alph + self.bet) -                     gammln(self.alph) -                     gammln(self.bet)
                
    def p(self,x):
        """Return probability density function."""
        if(x<=0.0 or x>=1.0): 
            raise Exception("bad x in Betadist")
        return math.exp((self.alph-1.0)*math.log(x) +                         (self.bet-1.0)*math.log(1.0-x)+self.fac)
    
    def cdf(self,x):
        """Return cumulative distribution function."""
        if(x<0.0 or x>1.0): raise Exception("bad x in Betadist")
        return self.betai(self.alph,self.bet,x)
    
    def invcdf(self,p):
        """Return inverse cumulative distribution function."""
        if(p<0.0 or p>1.0): raise Exception("bad p in Betadist")
        return self.invbetai(p,self.alph,self.bet)
                    


# In[ ]:




# ### Kolmogorov-Smirnov Distribution

# In[410]:

def invxlogx(y):
    """For negative y, 0>y>-e^(-1), return x such that y = xlog(x).
    The value returned is always the smaller of the two roots and
    is in the range 0 < x < e^(-1).
    """
    ooe = 0.33678879441171442322
    u=0.0
    t=0.0
    to=0.0
    if(y>=0.0 or y<= -ooe): raise Exception("no such inverse value")
    if(y<-0.2): u = math.log(ooe-math.sqrt(2.0*ooe*(y+ooe)))
    else: u = -10.0
    t=(math.log(y/u)-u)*(u/(1.0+u))
    u+=t
    if(t<1.0e-8 and abs(t+to)<0.01*abs(t)):
        to = t
        while(abs(t/u)>1.0e-15):
            t=(math.log(y/u)-u)*(u/(1.0+u))
            u+=t
            if(t<1.0e-8 and abs(t+to)<0.01*abs(t)): break
            to=t
    return math.exp(u)


# In[430]:

class KSdist(object):
    """Kolmogorow-Smirnov culative distribution function and their inverses."""
    def pks(self,z):
        """Return cumulative distribution function"""
        if(z<0.0): raise Exception("bad z in KSdist")
        if(z==0.0): return 0.0
        if(z<1.18):
            y = math.exp(-1.23370055013616983/(z**2))
            return 2.25675833419102515*math.sqrt(-math.log(y))                     *(y+(y**9)+(y**25)+(y*49))
        else:
            x = math.exp(-2.0*(z**2))
            return 1.0-2.0*(x-(x**4)+(x**9))
        
    def qks(self,z):
        """Return complementary cumulative distribution function."""
        if(z<0.0): raise Exception("bad z in KSdist")
        if(z==0.0): return 1.0
        if(z<1.18): return 1.0-self.pks(z)
        x = math.exp(-2.0*(z**2))
        return 2.0*(x-(x**4)+(x**9))
    
    def invqks(self,q):
        """Return inverse of the complementary cumulative distribution function."""
        x=0.0
        y=0.0
        if(q<=0.0 or q>1.0): raise Exception("bad q in KSdist")
        if(q==1.0): return 0.0
        if(q>0.3):
            f=-0.392699081698724155*((1.0-q)**2)
            y=invxlogx(f)
            yp = y
            logy = math.log(y)
            ff = f/((1.0+(y**4)+(y**12))**2)
            u = (y*logy-ff)/(1.0+logy)
            t=u/max(0.5,1.0-0.5*u/(y*(1.0+logy)))
            while( abs(t/y)>1.0e-15):
                f=-0.392699081698724155*((1.0-q)**2)
                y=invxlogx(f)
                yp = y
                logy = math.log(y)
                ff = f/((1.0+(y**4)+(y**12))**2)
                u = (y*logy-ff)/(1.0+logy)
                t=u/max(0.5,1.0-0.5*u/(y*(1.0+logy)))
            return 1.57079632679489662/math.sqrt(-math.log(y))
        else:
            x=0.03
            xp=x
            x = 0.5*q+(x**4)-(x**9)
            if(x>0.06): x+= (x**16)-(x**25)
            while( abs((xp-x)/x)>1.0e-15 ):
                xp=x
                x = 0.5*q+(x**4)-(x**9)
                if(x>0.06): x+= (x**16)-(x**25)
            return math.sqrt(-0.5*math.log(x))
    
    def invpks(self,p):
        """Return inverse of the cumulative distribution funciton."""
        return invqks(1.0-p)
            


# In[ ]:




# ### Poisson Distribution

# In[ ]:

class Poissondist(Gamma):
    """Poisson distribution, derived from the gamm funciton Gamma."""
    lam=0.0
    
    def __init__(self,llam):
        """Constructor. Initialize with lambda."""
        if( self.lam < 0.0):
            raise Exception("bad lam in Poissondist")
    
    def p(self,n):
        """Return probability density function."""
        if(n<0): raise Exception("bad n in Poissondist")
        return math.exp(-lam+n*math.log(lam)-gammln(n+1.0))
    
    def cdf(self,n):
        """Return cumulative distribution function."""
        if(n<0): raise Exception("bad n in Poissondist")
        if(n==0): return 0.0
        return self.gammq(n*1.0,lam)
    
    def invcdf(self,p):
        """Given argument P, return integer n 
        such that P(<n) <= P <= P(<n+1)
        """
        inc=1
        n=nl=nu=0
        if(p<=0.0 or p>=1.0): raise Exception("bad p in Poissondist")
        if(p<math.exp(-lam)): return 0
        n = int(max(math.sqrt(lam),5.0))
        if(p<self.cdf(n)):
            n=max(n-inc,0)
            inc *= 2
            while(p<cdf(n)):
                n=max(n-inc,0)
                inc *= 2
            nl = n
            nu = n + inc/2
        else:
            n += inc
            inc *= 2
            while(p>cdf(n)):
                n += inc
                inc *= 2
            nu = n
            nl = n - inc/2
        while(nu-nl>1):
            n = (nl+nu)/2
            if(p>cdf(n)): nu = n
            else: nl = n
        return nl


# In[ ]:




# ### Binomial Distribution

# In[412]:

class Bionialdist(Beta):
    """Binomial distribution, derived from the beta funciton Beta."""
    n=0
    pe=0.0
    fac=0.0
    
    def __init__(self,nn,ppe):
        """Constructor. Initialize with n (sample size) and
        p (event probability)."""
        self.n = nn
        self.pe = ppe
        if(self.n<=0 or self.pe<=0.0 or self.pe>=1.0):
            raise Exception("bad args in Binomialdist")
        self.fac = gammln(self.n+1.0)
        
    def p(self,k):
        """Return probability density function."""
        if(k<0): raise Exception("bad k in Binomialdist")
        if(k>n): return 0.0
        return math.exp(k*math.log(self.pe)+(n-k)*math.log(1.0-self.pe) +                         self.fac-gammln(k+1.0)-gammln(self.n-k+1.0))
        
    def cdf(self,k):
        """Return cumulative distribution function."""
        if(k<0): raise Exception("bad k in Binomialdist")
        if(k==0): return 0.0
        if(k>self.n): return 1.0
        return 1.0-self.betai(k,self.n-k+1.0,self.pe)
    
    def invcdf(self,p):
        """Given argument P, return integer n such that
        P(<n) <= P <= P(<n+1)
        """
        k=kl=ku=inc=1
        if(p<=0.0 or p>=1.0):
            raise Exception("bad p in Binomialdist")
        k = max(0,min(self.n,int(self.n*self.pe)))
        if(p<cdf(k)):
            k = max(k-inc,0)
            inc *= 2
            while(p<cdf(k)):
                k = max(k-inc,0)
                inc *= 2
            kl = k
            ku = k + inc/2
        else:
            k = min(k+inc,self.n+1)
            inc *= 2
            while(p>cdf(k)):
                k = min(k+inc,self.n+1)
                inc *= 2
            kl = k
            ku = k - inc/2
        while(ku-kl>1):
            k = (kl+ku)/2
            if(p<cdf(k)): 
                ku = k
            else:
                kl = k
        return kl


# In[ ]:


