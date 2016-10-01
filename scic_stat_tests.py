
# # Statistical Description of Data

# In[1]:

#import the python file of Distribution Functions
import scic_dist_functions as scic_dist


# Out[1]:

#     3!= 6.0 = e^(ln[gamma(4)]) = 6
#     120.0
#     4.03291461127e+26
#     24.0
#     0.994376487882
#     0.00562351211827
#     1.0
#     0.112754082424
#     0.520357177501
#     0.842681743299
#     0.999999999998
#     -0.112754082424
#     -0.520357177501
#     -0.842681743299
#     -0.999999999998
# 

# In[2]:

# quick test that I can create objects
my_ksdist = scic_dist.KSdist()


# In[3]:

print my_ksdist.pks(1.0)
print my_ksdist.pks(2.0)


# Out[3]:

#     36.4981663726
#     0.999329074744
# 

# ## Student's t-Test for Significantly different Means

# In[12]:

def ttest(data1, data2):
    """Given the arrays data1[0..n1-1] and data2[0..n2-1], returns an array of Student's t as t
    and its p-value as prob, small values of prob indicating that the arrays have significantly 
    different means. The data arrays are assumed to be drawn from populations with the same
    true variance.
    
    The p-value is a number between 0 and 1. It is the probability that |t| could be this large
    or larger just by chance, for distributions with equal means.
    """
    beta = Beta()
    t = 0.0
    prob = 0.0
    
    n1=len(data1)
    n2=len(data2)
    [ave1,var1] = avevar(data1)
    [ave2,var2] = avevar(data2)
    df=n1+n2-2
    svar=((n1-1)*var1+(n2-1)*var2)/df
    t = (ave1-ave2)/math.sqrt(svar*(1.0/n1+1.0/n2))
    prob = beta.betai(0.5*df,0.5,df/(df+t*t))
    
    return [t,prob]


# In[13]:

def avevar(data):
    """Given array data[0..n-1], returns its mean as ave and its variance as var in an array.
    """
    s=ep=0.0
    n=len(data)
    ave=0.0
    var=0.0
    
    for j in xrange(0,n): ave += data[j]
    ave /= n
    
    for j in xrange(0,n):
        s=data[j] - ave
        ep += s
        var += s*s
        
    var = (var-ep*ep/n)/(n-1)
    return [ave,var]


# In[14]:

def tutest(data1,data2):
    """ Given the arrays data1[0..n1-1] and data2[0..n2-1], returns an array of Student's t as t
    and its p-value as prob, small values of prob indicating that the arrays have significantly 
    different means. The data arrays are allowed to be drawn from populations with unequal variances.
    """
    beta = Beta()
    n1=len(data1)
    n2=len(data2)
    [ave1,var1] = avevar(data1)
    [ave2,var2] = avevar(data2)
    t=(ave1-ave2)/math.sqrt(var1/n1 + var2/n2)
    df = ((var1/n1+var2/n2)**2)/(((var1/n1)**2)/(n1-1) + ((var2/n2)**2)/(n2-1))
    prob = beta.betai(0.5*df,0.5,df/(df+(t**2)))
    
    return [t,prob]


# In[15]:

def tptest(data1,data2):
    """Given paired arrays data1[0..n1-1] and data2[0..n2-1], this routine returns Student's t
    for paired data as t, and its p-value, small values of prob indicating a significant
    difference of means.
    """
    cov=0.0
    beta = Beta()
    n=len(data1)
    [ave1,var1] = avevar(data1)
    [ave1,var2] = avevar(data2)
    for j in xrange(0,n): cov += (data1[j]-ave1)*(data2[j]-ave2)
    df = n-1
    cov /= df
    sd = math.sqrt((var1+var2-2.0*cov)/n)
    t = (ave1-ave2)/sd
    prob = beta.betai(0.5*df,0.5,df/(df+t*t))
    
    return [t,prob]


# ## F-Test for Significantly Different Variances

# In[16]:

def ftest(data1,data2):
    """Given the arrays data1[0..n1-1] and data2[0..n2-1], this routine returns the value of f,
    and its p-value as prob. Small values of prob indicate that the two arrays have significantly
    different variances.
    """
    beta = Beta()
    f=df1=df2=0.0
    n1=len(data1)
    n2=len(data2)
    [ave1,var1] = avevar(data1)
    [ave1,var2] = avevar(data2)
    if(var1 > var2 ):
        f = var1/var2
        df1 = n1-1
        df2 = n2-1
    else:
        f = var2/var1
        df1 = n2-1
        df2 = n1-1
    prob = 2.0*beta.betai(0.5*df2, 0.5*df1, df2/(df2+df1*f))
    if(prob > 1.0): prob = 2.0 - prob
    
    return [f,prob]


# ## Chi-Square Test

# In[17]:

def chsone(bins,ebins, knstrn=1):
    """Given the array bins[0..nbins-1] containing the observed numbers of events, and an array 
    ebins[0..nbins-1] containing the expected numbers of events, and given the number of
    constraints knstrn (normally one), this routine returns (trivially) the number of degrees of 
    freedom df, and (nontrivially) the chi-square chsq and the p-value prob, all as an array. A 
    small value of prob indicates a significant difference between the distributions bins and 
    ebins. Note that bins and ebins are both double arrays, although bins will normally contain 
    integer values.
    """
    gam = Gamma()
    nbins = len(bins)
    df=nbins-knstrn
    chsq=prob=0.0
    for j in xrange(0,nbins):
        if(ebins[j]<0.0 or (ebins[j]==0.0 and bins[j]>0.0)):
            raise Exception("bad expected number in chsone")
        if(ebins[j]==0.0 and bins[j]==0.0):
            df -= 1
        else:
            temp = bins[j]-ebins[j]
            chsq += temp*temp/ebins[j]
    prob = gam.gammq(0.5*df,0.5*chsq)
    
    return [df, chsq, prob]


# In[20]:

def chstwo(bins1, bins2, knstrn=1):
    """Given the array bins1[0..nbins-1] and bins2[0..nbins-1], containing two sets of binned
    data, and given the number of constraints knstrn (normally 1 or 0), this routine returns the
    number of degrees of freedom df, the chi-square chsq, and the p-value prob, all as an array.
    A small value of prob inidcates a significant difference between the distributions bins1 and
    bins2. Note that bins1 and bins2 are both double arrays, although they will normally contain
    integer values.
    """
    gam = Gamma()
    nbins = len(bins)
    df=nbins-knstrn
    chsq=prob=0.0
    for j in xrange(0,nbins):
        if(bins1[j]==0.0 and bins2[j]==0.0):
            df -= 1
        else:
            temp = bins1[j]-bins2[j]
            chsq += temp*temp/(bins1[j]+bins2[j])
    prob=gam.gammq(0.5*df, 0.5*chsq)
    
    return [df, chsq, prob]


# ## Kolmogorov-Smirnov Test

# In[25]:

def ksone( data, func ):
    """Given an array data[0..n-1], and given a user-supplied function of a single variable func 
    that is a cumulative distribution function ranging from 0 (for smallest values of its argument) 
    to 1(for largest values of its argument), this routine returns the K-S statistic d and the p-value
    prob, all as an array. Small values of prob show that the cumulative distribution function of data
    is significantly different from func. The array data is modified by being sort into ascedning order.
    """
    n = len(data)
    fo = 0.0
    ks = KSdist()
    data.sort()
    en=n
    d=0.0
    for j in xrange(0,n):
        fn=(j+1)/en
        ff=func(data[j])
        dt=max(abs(fo-ff),abs(fn-ff))
        if(dt>d): d=dt
        fo=fn
    en=math.sqrt(en)
    prob=ks.qks((en+0.12+0.11/en)*d)
    
    return [d,prob]


# In[26]:

def kstwo(data1, data2):
    """Given an array data1[0..n1-1], and an array data2[0..n2-1], this routine returns the K-S
    statistic d and the p-vale prob for the null hypothesis that the data sets are drawn from the
    same distribution. Small values of prob show that the cumulative distribution function of data1
    is significantly different from the that of data2. The arrays data1 and data2 are modified by being
    sorted into ascending order.
    """
    j1=j2=0
    n1=len(data1)
    n2=len(data2)
    fn1=fn2=0.0
    ks = KSdist()
    data1.sort()
    data2.sort()
    en1=n1
    en2=n2
    d=0.0
    while(j1<n1 and j2<n2):
        d1=data1[j1]
        d2=data2[j2]
        if(d1<=d2):
            j1 += 1
            fn1 = j1/en1
            while(j1<n1 and d1==data1[j1]):
                j1 += 1
                fn1 = j1/en1
        if(d2<=d1):
            j2 += 1
            fn2 = j2/en2
            while(j2<n2 and d2==data2[j2]):
                j2 += 1
                fn2 = j2/en2
        dt=abs(fn2-fn1)
        if(dt>d): d = dt
    en=math.sqrt(en1*en2/(en1+en2))
    prob=ks.qks((en+0.12+0.11/en)*d)
    
    return [d,prob]


# ## Measure of Association Based on Chi-Sqaure

# In[40]:

def cntab(nn):
    """Given a two-dimensional contingency table in the form of an array nn[0..ni-1][0..nj-1] of
    integers, this routine returns the chi-square chisq, the number of degrees of freedom df, the
    p-value prob (small values indicating a significant association), and two measures of association,
    Cramer's V (cramrv) and the contingency coefficient C (ccc), all as an array.
    """
    TINY = 1.0e-30
    gam = Gamma()
    i=j=nnj=nni=minij=0
    ni = len(nn)
    nj = len(nn[1])
    excptd=temp=sm = 0.0
    sumi = range(0,ni)*0.0
    sumj = range(0,nj)*0.0
    nni=ni
    nnj=nj
    for i in xrange(0,ni):
        sumi[i]=0.0
        for j in xrange(0,nj):
            sumi[i] += nn[i][j]
            sm += nn[i][j]
        if( sumi[i] == 0.0 ): nni -= 1
    for j in xrange(0,nj):
        sumj[j]=0.0
        for i in xrange(0,ni): sumj[j] += nn[i][j]
        if( sumj[j] == 0.0): nnj -= 1
    df=nni*nnj-nni-nnj+1
    chisq=0.0
    for i in xrange(0,ni):
        for j in xrange(0,nj):
            expctd = sumj[j]*sumi[i]/sm
            temp=nn[i][j]-expcted
            chisq += temp*temp/(expctd+TINY)
    prob = gam.gammq(0.5*df,0.5*chisq)
    if( nni < nnj ): minij = nni - 1
    else: minij = nnj - 1
    cramrv=math.sqrt(chisq/(sm*minij))
    ccc=math.sqrt(chisq/(chisq+sm))


# ## Linear Correlation

# In[42]:

def pearsn(x,y):
    """Given two arrays x[0..n-1] and y[0..n-1], this routine computes their correclation coefficient r 
    (returned as r), the p-value at which the null hypothesis of zero correlation is disproved (prob 
    whose small value indicates a significant correlation), and Fisher's z (return as z). whose value
    van be used in further statistical tests. r, prob and z are returned as an array.
    """
    TINY = 1.0e-20
    beta = Beta()
    n=len(x)
    yt=xt=t=df=syy=sxy=sxx=ay=ax=0.0
    for j in xrange(0,n):
        ax += x[j]
        ay += y[j]
    ax /= n
    ay /= n
    for j in xrange(0,n):
        xt=x[j]-ax
        yt=y[j]-ay
        sxx += xt*xt
        syy += yt*yt
        sxy += xt*yt
    r = sxy/(math.sqrt(sxx*syy)+TINY)
    z = 0.5*math.log((1.0+r+TINY)/(1.0-r+TINY))
    df = n-2
    t = r*math.sqrt(df/((1.0-r+TINY)*(1.0+r+TINY)))
    prob = beta.betai(0.5*df, 0.5, df/(df+t*t))
    
    return [r,prob,z]


# ## Nonparametric Correlation

# ### Spearman Rank-Order Correlation Coefficient

# In[67]:

def spear(data1, data2):
    """Given two data arrays, data1[0..n-1] and data2[0..n-1], this routine returns their sum 
    squared difference of ranks as d, the number of standard deviations by which d deviates from
    its null-hypothesis expected value as zd, the two-sided p-value of this deviation as probd,
    Spearman's rank correlation r_s as rs, and the two sided p-value of its deviation from 0 as
    probrs, all as an array. The external routines  crank and sort2 are used. A small value of either
    probd or probrs indicates a significant correlation (rs positive) or anticorrelation (rs negative).
    """
    d=zd=probd=rs=probrs=0.0
    bet = Beta()
    n = len(data1)
    vard=t=sg=sf=fac=en3n=en=df=aved
    wksp1 = range(n)*0.0
    wksp2 = range(n)*0.0
    [wksp1,wksp2] = sort2(data1,data2)
    [wksp1, sf ] = crank(wksp1)
    [wksp2, wksp1] = sort2(wksp2, wksp1)
    [wksp2, sg ] = crank(wksp2)
    d=0.0
    for j in xrange(n):
        d += (wksp1[j]-wksp2[j])**2
    en = n
    en3n = en*en*en-en
    aved = en3n/6.0-(sf+sg)/12.0
    fac=(1.0-sf/en3n)*(1.0-sg/en3n)
    vard = ((en-1.0)*en*en*((en+1.0)**2)/36.0)*fac
    zd = (d-aved)/math.sqrt(vard)
    probd = erfcc(abs(zd)/1.4142136)
    rs=(1.0-(6.0/en3n)*(d+(sf+sg)/12.0))/math.sqrt(fac)
    fac=(rs+1.0)*(1.0-rs)
    if(fac > 0.0):
        t = rs*math.sqrt((en-2.0)/fac)
        df = en-2.0
        probrs = bet.betai(0.5*df,0.5,df/(df*t*t))
    else:
        probrs=0.0
        
    return [d,zd,probd,rs,probrs]


# In[68]:

def sort2(data1,data2):
    """Sort an array data1[0..n-1] into ascending order, while making the corresponding 
    rearrangement of the array data2[0..n-1].
    """
    n = len(data1)
    srt = sorted(range(n), key=data1.__getitem__)
    wksp1 = range(n)*0.0
    wksp2 = range(n)*0.0
    i = 0
    for j in srt:
        wksp1[i] = data1[j]
        wksp2[i] = data2[j]
    return [ wksp1, wksp2 ]


# In[70]:

def crank(wdata):
    """Given a sorted array w[0..n-1], replaces the elements by their rank, including midranking
    of ties, and returns as s the sum of f^3 - f, where f is the number of elements in each tie.
    """
    j=1
    ji=jt=0
    w = list(wdata)
    n=len(w)
    t=rank=s=0.0
    while(j<n):
        if( not (w[j] == w[j-1])):
            w[j-1] = j
            j += 1
        else:
            jt = j+1
            while(jt<=n and w[jt-1]==w[j-1]):
                jt += 1
            rank = 0.5*(j+jt+1)
            for ji in xrange(j,(jt)):
                w[ji-1]=rank
            t = jt-j
            s += (t*t*t-t)
            j = jt
    if(j==n): w[n-1]=n
        
    return [w,s]


# ## Kendall's Tau

# In[72]:

def kendl1(data1, data2, tau):
    """Given data arrays data1[0..n-1] and data2[0..n-1], this program return Kendall's tau as
    tau, its number of standard deviations frmo zero as z, and its two side p-value as prob. Small
    values of prob indicate a significant correlation (tau positive) or anitcorrelation( tau 
    negative).
    """
    tau=z=prob=0.0
    iss=n2=n1=0
    n = len(data1)
    for j in xrange(n-1):
        for k in xrange(j+1,n):
            a1 = data1[j]-data1[k]
            a2 = data2[j]-data2[k]
            aa=a1*a2
            if( not aa == 0.0 ):
                n1 += 1
                n2 += 2
                if( aa > 0.0): iss += 1
                else: iss -= 1
            else:
                if( not a1 == 0.0): n1 += 1
                if( not a2 == 0.0): n2 += 1
    tau = iss/( math.sqrt(n1)*math.sqrt(n2))
    svar=(4.0*n+10.0)/(9.0*n*(n-1.0))
    z=tau/math.sqrt(svar)
    prob=erfcc(abs(z)/1.4142136)
    
    return [tau,z,prob]


# In[73]:

def kendl2(tab):
    """Given a two-dimensional table tab[0..i-1][0..j-1], such that tab[k][l] contains the 
    number of events falling in bin k of on variable and bin l of another, this program returns
    Kendall's tau as tau, its number of standard deviations from zero as z, and its two-sided
    p-value as prob, all as an array. Small values of prob indicate a significant correlation
    (tau positive) or anticorrelation (tau negative) between the two variables. Although tab
    is a double array, it will normally contain integral values.
    """
    tau=z=prob=0.0
    s=en1=en2=points=svar=0.0
    i = len(tab)
    j = len(tab[1])
    nn = i*j
    points=tab[i-1][j-1]
    for k in xrange(nn-1):
        ki=(k/j)
        kj=k-j*ki
        points += tab[ki][kj]
        for l in xrange(k+1,nn):
            li=l/j
            lj=l-j*li
            m1=li-ki
            m2=lj-kj
            mm=m1*m2
            pairs=tab[ki][kj]*tab[li][lj]
            if( not mm == 0):
                en1 += pairs
                en2 += pairs
                if( mm > 0): s += pairs
                else: s -= pairs
            else:
                if( not m1 == 0): en1 += pairs
                if( not m2 == 0): en2 += pairs
    tau=s/math.sqrt(en1*en2)
    svar=(4.0*points+10.0)/(9.0*points*(points-1.0))
    z=tau/math.sqrt(svar)
    prob=erfcc(abs(z)/1.4142136)
    
    return [tau,z,prob]


# ## Two-Dimensional K-S

# In[77]:

def ks2d1s(x1,y1,quadvl):
    """Two-dimensional Kolmogorov-Smirnov test of one sample against a model. Given the x and
    y coordinates of n1 data points in arrays x1[0..n1-1] and y1[0..n1-1], and given a user-
    supplied function quadvl that exemplifies the model, this routine returns the two-dimensional
    K-S statistic as d1, and its p-value as prob, all as an array. Small values of prob show that
    the sample is significantly different from the model. Note that the test is slightly distribution-
    dependent, so prob is only an estimate.
    """
    d1=prob=0.0
    n1=len(x1)
    r1=dum=dumm=0.0
    ks = KSdist()
    for j in xrange(n1):
        [fa,fb,fc,fd] = quadct(x1[j],y1[j],x1,y1)
        [ga,gb,gc,gd] = quadvl(x1[j],y1[j])
        if(fa>ga): fa += 1.0/n1
        if(fb>gb): fb += 1.0/n1
        if(fc>gc): fc += 1.0/n1
        if(fd>gd): fd += 1.0/n1
        d1 = max(d1,abs(fa-ga))
        d1 = max(d1,abs(fb-gb))
        d1 = max(d1,abs(fc-gc))
        d1 = max(d1,abs(fd-gd))
    [r1,dum,dumm] = pearsn(x1,y1)
    sqen = math.sqrt(n1)
    rr=math.sqrt(1.0-r1*r1)
    prob = ks.qks(d1*sqen/(1.0+rr*(0.25-0.75/sqen)))
    
    return [d1,prob]


# In[76]:

def quadct(x,y,xx,yy):
    """Given an origin (x,y), and an array of nn points with coordinates
    xx[0..nn-1] and yy[0..nn-1], count how many of them are in each
    quadrant around the origin, and return the normalized fractions.
    Quadrants are labeled alphabetically, a counterclockwise from the 
    upper right. Used by ks2d1s and ks2d2s.
    """
    na=nb=nc=nd=0
    nn = len(xx)
    for k in xrange(nn):
        if(yy[k]==y and xx[k]==x): continue
        if(yy[k]>y):
            if(xx[k]>x): na += 1
            else: nb += 1
        else:
            if(xx[k]>x): nd += 1
            else: nc += 1
    ff=1.0/nn
    return [ff*na,ff*nb,ff*nc,ff*nd]


# In[79]:

def quadvl(x,y):
    """This is a sample of a user-supplied routine to be used with ks2d1s.
    In this case, the model distribution is uniform inside the square.
    """
    qa=min(2.0,max(0.0,1.0-x))
    qb=min(2.0,max(0.0,1.0-y))
    qc=min(2.0,max(0.0,x+1.0))
    qd=min(2.0,max(0.0,y+1.0))
    fa=0.25*qa*qb
    fb=0.25*qb*qc
    fc=0.25*qc*qd
    fd=0.25*qd*qa
    
    return [fa,fb,fc,fd]


# In[80]:

def ks2d2s(x1,y1,x2,y2):
    """Two-dimensional Kolmogorow-Smirnov test on two sampls. Given the x
    and y coordinates of the first sample as n1 values in arrays x1[0..n1-1]
    and y1[0..n1-1], and likewise for the second sample, n2 values in arrays
    x2 and y2, this routine returns the two-dimensional, two-sample K-S 
    statistic as d, and its p-value as prob, all as an array. Small values 
    of prob show that the two samples are significantly different. Note that
    the test is slightly distribution-dependent, so prob is only an estimate.
    """
    n1=len(x1)
    n2=len(x2)
    r1=r2=rr=dum=dumm=0.0
    ks = KSdist()
    d1=0.0
    for j in xrange(n1):
        [fa,fb,fc,fd] = quadct(x1[j],y1[j],x1,y1)
        [ga,gb,gc,gd] = quadct(x1[j],y1[j],x2,y2)
        if(fa>ga): fa += 1.0/n1
        if(fb>gb): fb += 1.0/n1
        if(fc>gc): fc += 1.0/n1
        if(fd>gd): fd += 1.0/n1
        d1 = max(d1,abs(fa-ga))
        d1 = max(d1,abs(fb-gb))
        d1 = max(d1,abs(fc-gc))
        d1 = max(d1,abs(fd-gd))
    d2=0.0
    for j in xrange(n2):
        [fa,fb,fc,fd] = quadct(x2[j],y2[j],x1,y1)
        [ga,gb,gc,gd] = quadct(x2[j],y2[j],x2,y2)
        if(ga>fa): ga += 1.0/n1
        if(gb>fb): gb += 1.0/n1
        if(gc>fc): gc += 1.0/n1
        if(gd>fd): gd += 1.0/n1
        d2 = max(d2,abs(fa-ga))
        d2 = max(d2,abs(fb-gb))
        d2 = max(d2,abs(fc-gc))
        d2 = max(d2,abs(fd-gd))
    d=0.5*(d1+d2)
    sqen=math.sqrt(n1*n2/float(n1+n2))
    [r1,dum,dumm] = pearsn(x1,y1)
    [r2,dum,dumm] = pearsn(x2,y2)
    rr = math.sqrt(1.0-0.5*(r1*r1+r2*r2))
    prob=ks.qks(d*sqen/(1.0+rr*(0.25-0.75/sqen)))
            


# ## Savitzky-Golay Smoothing Filters
# 
# I'm not going to implement this algorithm yet but I will write it out.

# In[91]:

def savgol(cc,np,nl,nr,ld,m):
    """Returns in c[0..np-1], in wraparound order (N.B.!) consistent with the
    argument respns in routine convlv, a set of Savitzky-Golay filter 
    coefficients. nl is the number of leftward (past) data points used,
    while nr is the number of rightward (future) data points, making the 
    total number of points used nl + nr + 1. ld is the order of the
    derivative desired (e.g., ld = 0 for smoothed funciton. For the 
    derivative of k, you must mulitply the array c by k!.) m is the order
    of the smoothing polynomial, also equal to the highest conserved
    moment; usual values are m = 2 or m = 4.
    """
    c = list(cc) # make a copy of cc
    j=k=imj=ipj=kk=mm=0
    fac=summ=0.0
    if( np<nl+nr+1 or nl<0 or nr<0 or ld>m or nl+nr<m ):
        raise Exception("bad args in savgol")
    
    a = [ [0.0 for x in xrange(m+1)] for y in xrange(m+1)]
    b = [ 0.0 for x in xrange(m+1)]
    for ipj in xrange((m << 1)+1):
        if(not ipj == 0): summ = 0.0
        else: summ = 1.0
        for k in xrange(1,nr+1): summ += (float(k)**float(ipj))
        for k in xrange(1,nl+1): summ += ((-floa(k))**float(ipj))
        mm=min(ipj,2*m-ipj)
        for imj in xrange(-mm,mm+1,2):
            a[(ipj+imj)/2][(ipj-imj)/2]=summ
    alud = LUdcmp(a)
    for j in xrange(m+1): b[j]=0.0
    b[ld]=1.0
    alud.solve(b,b)
    for kk in xrange(np):
        c[kk]=0.0
    for k in xrange(-nl,nr+1):
        summ=b[0]
        fac=1.0
        for mm in xrange(1,m+1):
            fac *= k
            summ += b[mm]*fac
        kk=(np-k)%np
        c[kk]=summ
    
    return c


# In[ ]:


