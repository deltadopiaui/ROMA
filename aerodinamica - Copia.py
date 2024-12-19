import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from pathlib import Path
import os

                 #'nome':       [   a0,   Cl0, Clmax,   Cmac, t/c_max,  x/c,    h0]
AEROFOLIO_DICT = {'s1223':      [ 5.98, 1.171, 2.337, -0.286,   0.121, 0.200,  0.236], #dados análise xflr5
                  'e205i':      [ 5.73,-0.257, 0.415,  0.084,   0.105, 0.297,  0.207], #dados análise xflr5
                  'de4-77':     [ 6.08,	0.978, 2.55,  -0.250,	0.138, 0.222,  0.260], #dados do ensaio com 10 medidas por modelo e 1 modelo. Data do ensaio 19/06/2020
                  'ds3_027':    [ 6.14,	0.970, 2.55,  -0.257,   0.141, 0.171,  0.269], #dados do ensaio com 10 medidas por modelo e 1 modelo. Data do ensaio 19/06/2020
                  'ds3-023_flap':    [ 6.14,	0.970, 2.55,  -0.257,   0.141, 0.171,  0.269],
                  'n10i_0':          [ 5.73, -0.61, 0.41, 0.086,   0.113, 0.264,  0.25],
                  'n10i_pos':          [ 5.73, -0.61, 0.41, 0.086,   0.113, 0.264,  0.25],
                  'n10i_neg':          [ 5.73, -0.61, 0.41, 0.086,   0.113, 0.264,  0.25],
                  'de4-77_flap':    [6.08,	0.960, 2.55,  -0.250,	0.138, 0.222,  0.260],
                  'S1223':          [5.79, 1.1773, 2.2179, -0.27, 0.212, 0.198, 0.250],
                   'S1223_flap':   [5.79, 1.1773, 2.2179, -0.27, 0.212, 0.198, 0.250],
                   'e423_inv':      [4.73, -1.098, 0.3213, 0.236, 0.125, 0.237, 0.250],
                   's1223_flap':      [ 5.98, 1.171, 2.337, -0.286,   0.121, 0.200,  0.236],
                   'DF102_inv':[4.75, -0.36, 0.36, 0.05, 0.11, 0.291, 0.25],
                   'twist_wing':[3.88,1.141,2.25,-0.23,0.138,0.171,0.26],
                   'torc_geo':[3.84,1.12,2.23,-0.27,0.1405,0.3083,0.25]

                  }

def ler(arquivo, linhas_pular, linhas_ler=None):
    with open(arquivo,'r') as f:
        x=np.loadtxt(f,dtype=float,skiprows=linhas_pular,unpack=True, max_rows = linhas_ler)
    return np.array(x).T

def rotaciona_referencial(x,y,dtheta):
    x_novo = x*np.cos(dtheta) + y*np.sin(dtheta)
    y_novo = x*np.sin(dtheta) - y*np.cos(dtheta) 
    return abs(x_novo), abs(y_novo)

def polar_3(x,a0,a1,a2):
  f =  a0 + a1*x + a2*x**2 
  return f

def extrai_asa(alfa_asa, CL_asa, CD_asa):

    popt, pcov = opt.curve_fit(polar_3, CL_asa, CD_asa)
    CD0_asa, CD1_asa, CD2_asa = popt
    CLo_asa = CL_asa[np.where(alfa_asa == 0)]
    CLmax_asa = CL_asa[-1]
    a_asa = (CLmax_asa-CLo_asa)/np.radians(alfa_asa[-1]-0)
    alfa_sn_asa=-CLo_asa/a_asa

    return CD0_asa, CD1_asa, CD2_asa, CLo_asa, CLmax_asa, a_asa, alfa_sn_asa

def ler_txt(arquivo,linhas_pular):
    with open(arquivo,'r') as f:
        x=np.loadtxt(f,dtype=float,skiprows=linhas_pular,unpack=True)
    return np.array(x)

def calcula_ea(lht, ht, hpn, A, S, CL = 0, a = 0, theta = 0):
    
    if CL*a == 0:
        metodo = 'simples'
    else:
        metodo = 'detalhado'
  
    def simples(lht, ht, hpn, A, S, CL, a, theta):
    
        hH = ht-hpn
        b = (A/S)**0.5
        Kh = (1-hH/b)/(2*lht/b)**0.5
        Ka = 1/A - 1/(1+A**1.7)
        ea = 4.44*(Ka*Kh)**1.19
        
        return ea
        
    def detalhado(lht, ht, hpn, A, S, CL, a, theta):
        
        (X, h_d) = rotaciona_referencial(lht, ht-hpn, theta)
        s = 0.5*(S*A)**0.5
        X_s = X/s

        cwd = os.getcwd()
        
        invA_vec = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6b/d_sCL-eixoinvA.txt',0)
        lambd_vec = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6b/d_sCL-eixolambd.txt',0)
        X_s_vec = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6b/d_sCL-eixoX_s.txt',0)
     
        d_sCL_vec = np.zeros(np.shape(invA_vec)[0]*np.shape(X_s_vec)[0]*np.shape(lambd_vec)[0])
        d_sCL_vec = d_sCL_vec.reshape(np.shape(invA_vec)[0],np.shape(lambd_vec)[0],np.shape(X_s_vec)[0])
        
        linha = 0
       
        for i in range(np.shape(invA_vec)[0]):
        
            d_sCL_vec[i] = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6b/d_sCL.txt',linha, np.shape(lambd_vec)[0])
            linha += np.shape(lambd_vec)[0] + 1
        
        d_sCL_func = interpolate.RegularGridInterpolator((invA_vec, lambd_vec, X_s_vec), d_sCL_vec, method='nearest', bounds_error=False, fill_value=None)
        
        d_sCL = d_sCL_func(np.array([1/A, 1, X_s]))[0]
        
        d = d_sCL*(s*CL)
        
        h = h_d + d
        
        h_s = h/s
        
        invA_vec = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6c/e_CL-eixoinvA.txt',0)
        h_s_vec = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6c/e_CL-eixoh_s.txt',0)
        X_s_vec = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6c/e_CL-eixoX_s.txt',0)
     
        e_CL_vec = np.zeros(np.shape(invA_vec)[0]*np.shape(X_s_vec)[0]*np.shape(h_s_vec)[0])
        e_CL_vec = e_CL_vec.reshape(np.shape(invA_vec)[0],np.shape(X_s_vec)[0],np.shape(h_s_vec)[0])
        
        linha = 0
        
        for i in range(np.shape(invA_vec)[0]):
        
            e_CL_vec[i] = ler(str(Path.cwd())+'/BancoDeDados/downwash/pullin/fig4.6c/e_CL.txt',linha, np.shape(X_s_vec)[0])
            linha += np.shape(X_s_vec)[0] + 1
        
        e_CL_func = interpolate.RegularGridInterpolator((invA_vec, X_s_vec, h_s_vec), e_CL_vec, method='linear', bounds_error=False, fill_value=None)
        
        e_CL = e_CL_func(np.array([1/A, X_s, h_s]))[0]
        e_CL = e_CL*np.pi/180
        ea = e_CL*a

        return ea
    
    _metodos_={'simples':simples,
               'detalhado':detalhado}
    
    return  _metodos_[metodo](lht, ht, hpn, A, S, CL, a, theta)

class Asa():

    def __init__(self, Perfil_asa, AR, c,sweep_angle=0, conf_winglet = 'c0', metodo = 'arquivo'):
        self.sweep_angle = sweep_angle
        self.AR = AR
        self.c = c
        self.metodo = metodo
        self.conf_winglet = conf_winglet
        
        def perfil(Perfil_asa,sweep_angle):
            # beta = 1
            # b_tg_sweep = beta + np.tan(np.radians(sweep_angle))**2 
            [ a0, Cl0, Clmax, CM, t_c, x_c, pn] = AEROFOLIO_DICT[Perfil_asa]
            # k=a0/(2*np.pi)
            # a=2*np.pi*self.AR/(2+np.sqrt((self.AR**2/k**2)*b_tg_sweep+4))
            # CLmax = a*Clmax/a0
            a, CD0, CD1, CD2, CL0 = None, None, None, None, None
            return a0, Cl0, Clmax, CM, t_c, x_c, pn, Clmax, a, CD0, CD1, CD2, CL0
        
        def arquivo(Perfil_asa):
            
            arquivo_sufix = '_{conf}.txt'.format(conf=self.conf_winglet)
            if self.conf_winglet == 'c0':
                arquivo_sufix = '.txt'
            
            alfa_asa, CL_asa, CD_asa, CM, pn = ler_txt(str(Path.cwd())+'/BancoDeDados/polares/polares_'+Perfil_asa+arquivo_sufix,2)[0], ler_txt(str(Path.cwd())+'/BancoDeDados/polares/polares_'+Perfil_asa+arquivo_sufix,2)[1], ler_txt(str(Path.cwd())+'/BancoDeDados/polares/polares_'+Perfil_asa+arquivo_sufix,2)[2], ler_txt(str(Path.cwd())+'/BancoDeDados/polares/polares_'+Perfil_asa+arquivo_sufix,2)[3], ler_txt(str(Path.cwd())+'/BancoDeDados/polares/polares_'+Perfil_asa+arquivo_sufix,2)[4]
            CD0, CD1, CD2, CL0, CLmax, a, alfa_sn_asa  = extrai_asa(alfa_asa, CL_asa, CD_asa)
            
            a0, Cl0, Clmax = None, None, None
            [ a0_no, Cl0_no, Clmax_no, CM_no, t_c, x_c, pn_no] = AEROFOLIO_DICT[Perfil_asa]

            return a0, Cl0, Clmax, CM[0], t_c, x_c, pn[0], CLmax, a[0], CD0, CD1, CD2, CL0[0]
            
        dispatch_asa = {'perfil':perfil,
                        'arquivo':arquivo}
            
        self.a0, self.Cl0, self.Clmax, self.CM,  self.t_c, self.x_c, self.pn, self.CLmax, self.a, self.CD0, self.CD1, self.CD2, self.CL0 = dispatch_asa[metodo](Perfil_asa)
 
    def calcula_a(self):
    
        def perfil(self):
            k=self.a0/(2*np.pi)
            a=2*np.pi*self.AR/(2+np.sqrt((self.AR/k)**2+4))
            return a
        
        def arquivo(self):
     
            return self.a
        
        dispatch_asa = {'perfil':perfil,
                        'arquivo':arquivo}
        a = dispatch_asa[self.metodo](self)
        return a
    
    def calcula_CD(self, CL): 
    
        def perfil(self, CL):
            RE = 1.225*10*self.c/1.87e-5
            cf = 0.455/( ( (1+0.144*0.03**2)**(0.65) )*( np.log10(RE)**(2.58) ) )
            Sww_S = 1.977 + 0.52*self.t_c
            FF = ( 1 + (0.6/self.x_c)*self.t_c + 100*self.t_c**4 )*( 1.34*(0.03)**0.18 ) 
        
            CD0 = cf*FF*1.05*Sww_S
        
            e = 1.78*( 1-0.045*self.AR**0.68 )-0.64
            k = 1/( np.pi*self.AR*e )
        
            CD = CD0 + k*(CL**2)
       
            return CD
            
        def arquivo(self, CL):
     
            return self.CD0 + CL*self.CD1 + (CL**2)*self.CD2
        
        dispatch_asa = {'perfil':perfil,
                        'arquivo':arquivo}
        CD = dispatch_asa['arquivo'](self, CL)
        return CD

    def calcula_CL(self, i, a, alpha=0, epsilon=0, lt_RB=0, V=1, theta_dot=0) :
        
        def perfil(self, i, a, alpha, epsilon, lt_RB, V, theta_dot) :
            Clp=i*self.a0 + self.Cl0
            CL=(a/self.a0)*Clp + a*(alpha + lt_RB*theta_dot/V - epsilon)
            return CL
    
        def arquivo(self, i, a, alpha, epsilon, lt_RB, V, theta_dot) :
            CL= self.CL0  + a*(i + alpha + lt_RB*theta_dot/V - epsilon)
            return CL
        dispatch_asa = {'perfil':perfil,
                        'arquivo':arquivo}
        CL = dispatch_asa[self.metodo](self, i, a, alpha, epsilon, lt_RB, V, theta_dot)
        return CL
    
    def calcula_CLmax(self,a):
    
        def perfil(self,a):
            return (a/self.a0)*self.Clmax
        
        def arquivo(self,a):
            return self.CLmax
            
        dispatch_asa = {'perfil':perfil,
                        'arquivo':arquivo}
        CLmax = dispatch_asa[self.metodo](self, a)
        return CLmax
 
    def calcula_pn(self):
    
        pn = 0.0486*np.log(self.AR) + 0.1652 + self.pn - 0.25
        
        return pn
