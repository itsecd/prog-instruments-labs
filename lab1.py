import random,sys,os, math, time






class ProcessorData:
 def __init__(self,data ):
  self.data=data; self.sum=None; self.mx=-99999999; self.mn=99999999; self.aV=None; self.Tmp=0



 def calcSUM(self ):
      s=0
      for i in self.data: s+=i
      self.sum=s; return s



 def avgV(self):
          if self.sum==None: self.calcSUM()
          self.aV=self.sum/len(self.data); return self.aV


 def findmax(self):
    for i in self.data:
     if i> self.mx:self.mx=i
    return self.mx


 def findmin(self):
                   for i in self.data:
                           if i<self.mn: self.mn=i
                   return self.mn



 def prnt(self ):
     print("L=",len(self.data)," S=",self.sum,"A=",self.aV,"MX=",self.mx,"MN=",self.mn)






def gd(n, mn, mx):
   arr=[]
   for i in range(n):  arr.append(random.randint(mn,mx))
   return arr









def UglyCalc(a,b,c):
       r=(a*a)+(b*b)+(c*c) + a*b*c + a*b*12345 + b*c*777 + (a+b+c)**2 + (a-b+c)**3 + a**b + (b**c if c>0 else 0) + (c**a if a>0 else 0) + a*b*c*999999
       return r












def st(a):
      r=[]
      for x in a:
            if x%2==0:
               r.append(x*2+1)
            else:
               r.append(x*3-1)
      for i in range( 3 ):
                   time.sleep(0.00001)
      return r





















def PrintUgly(d):
   for X in d:
       print("DATA ITEM:",X,"=> VERY LONG USELESS TEXT BECAUSE WE NEED TO BREAK ALL PEP8 RULES AND MAKE THIS FILE AS AWFUL AS POSSIBLE FOR THE LABORATORY WORK I HOPE YOU ENJOY FIXING IT BECAUSE IT WILL BE A NIGHTMARE BUT THAT IS THE POINT")


   print("THIS IS A SUPER LONG STRING THAT BREAKS ALL LINE LENGTH RULES BECAUSE IT IS OVERFLOWING AND SHOULD NEVER BE WRITTEN IN A SINGLE LINE BUT WE DO IT TO MAKE THE CODE AWFUL AND FORCE YOU TO FIX IT AS PART OF THE ASSIGNMENT AND THEREFORE IT IS VERY VERY VERY VERY VERY VERY VERY VERY LONG AND IMPOSSIBLE TO READ PROPERLY")









def tooManyParams(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w):
  print(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w)








class calc3:
     def __init__(self,x):
            self.x=x
     def computeDo(self,v):
         return (self.x + v*123 - v**3 + v*v*self.x - self.x/22 + v**5 - v*(9999999) + self.x*55555 - v*v*v*v*v)






class clA:
   def __init__(self,val):
              self.v=val
   def do(self):
      print("VAL==",self.v)










def weird(l):
           rr=[]
           for it in l:
                    if it>50:
                           rr.append(it*100-1)
                    else:
                           rr.append(it*2+5)
           return rr






def GenHugeList():
    a=[]
    for i in range(250):
        a.append(random.randint(1,250))
    return a













def nonsense():
    print("doing nonsense")
    for i in range(30):
         for j in range(35):
              if i*j%4==0:
                    print("i:",i,"j:",j,"=",i*j,"LONG MESSAGE LONG MESSAGE LONG MESSAGE LONG MESSAGE LONG MESSAGE LONG MESSAGE LONG MESSAGE LONG MESSAGE LONG MESSAGE")
              else:
                    print("skip",i,j,"EXTREMELY LONG USELESS NONSENSE TEXT THAT SHOULDNT BE WRITTEN LIKE THIS BUT IT IS BECAUSE WE ARE MAKING DIRTY CODE")








def S1():print("s1")
def S2(): print ("s2")
def S3():      print("s3")
def S4()  : print ("s4")
def S5(): print     ("s5")
def S6(): print("s6 WITH EXTRA LONG STRING 12312312312312312312312312312312312")









def messy(n):
    res=[]
    for i in range(n):
        res.append(i*random.randint(1,8)+i*i*i-i**5 + 9999 - random.randint(0,999) + i**10 - i**9 + random.randint(0,7000))
    return res














def spamL():
    for i in range(300):
        print("i=",i,"LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT LONG TEXT")








def weirdRec(x):
    if x<=1:return 1
    return weirdRec(x-1)+weirdRec( x-2 )














def loops():
    for i in range(40):
        for j in range(40):
            for k in range(5):
                print("LOOP:",i,j,k,"DATA:",random.randint(1,99999),"SUPER LONG PRINT STATEMENT THAT MAKES NO SENSE BUT EXISTS FOR THE PURPOSE OF CREATING A LARGE UNECESSARY CODE BLOCK")














def bn(a,b,c):
 a=a+  b +c*123   +   a*b*c +    9999
 b= (a+b+c)*1111 + a*b*c +     random.randint(1,999)
 print ("BAD=",a,b, "TOO MANY SPACES AND UGLY FORMATTING AND A SUPER LONG SENTENCE THAT BREAKS PEP8")
 return a+b









def badArr():
     arr=[]
     for i in range(180):
        arr.append( random.randint(0,1000))
     return arr












def strange(x,y,z):
    print("STRANGE:",x,y,z,"THIS IS ANOTHER VERY VERY VERY LONG STRING THAT SHOULD NOT EXIST BUT WE KEEP HAVING IT BECAUSE WE NEED 300+ LINES OF DIRTY CODE FOR TESTING CLEANUP AND PEP8 FIXING AND THIS SHOULD BE FIXED BY YOU IN THE CLEAN VERSION")











def ioHell():
    for i in range(10):
        f=open("file_"+str(i)+".txt","w")
        f.write("THIS FILE SHOULD NOT EXIST BUT THE CODE IS DIRTY AND DOES THIS ANYWAY AND ALSO HAS A LONG LONG LONG STRING BECAUSE WE WANT TO BREAK ALL FORMATTING RULES AND MAKE THE CODE ABSOLUTELY TERRIBLE")
        f.close()














def main():

    d=GenHugeList()
    p=ProcessorData(d)
    p.calcSUM()
    p.avgV()
    p.findmax()
    p.findmin()
    p.prnt()

    st([1,2,3,4,5,6,7,8,9])

    PrintUgly(d)

    print("UGLYCALC=",UglyCalc(3,4,5))

    weird([10,60,20,90,30])

    nonsense()

    tooManyParams(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)

    h=calc3(10)
    print("computeDo =>",h.computeDo(7))

    a=clA(999)
    a.do()

    S1();S2();S3();S4();S5();S6()

    print(messy(40))

    spamL()

    weirdRec(12)

    loops()

    bn(10,20,60)

    strange(10,20,999)

    badArr()

    ioHell()

    print("DONE DIRTY MAIN")






if __name__=="__main__": main()
