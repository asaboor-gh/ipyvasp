import re
import os
from io import StringIO
from itertools import islice, chain, product
from collections import namedtuple
import textwrap
import xml.etree.ElementTree as ET

import numpy as np

from . import utils, serializer
#from ipyvasp import utils, serializer


def dict2tuple(name,d):
    """Converts a dictionary (nested as well) to namedtuple, accessible via index and dot notation as well as by unpacking.
    
    Parameters
    ----------
    name : str
        Name of the namedtuple
    d : dict
        Dictionary to be converted to namedtuple. It can be nested as well.
    """
    return namedtuple(name,d.keys())(
           *(dict2tuple(k.upper(),v) if isinstance(v,dict) else v for k,v in d.items())
           )
    
class DataSource:
    "Base class for all data sources. It provides a common interface to access data from different sources. Subclass it to get data from a source."
    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("File: '{}'' does not exist!".format(path))
        self._path = path
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.path!r})"
    
    @property
    def path(self): return self._path
    
    @property
    def bands(self):
        "Returns a Bands object to access band structure data and plotting methods."
        from .api import Bands
        return Bands(self)
    
    def get_efermi(self, evals, occs, tol = 1e-3):
        "Gets Fermi energy or VBM from evals and occs."
        if np.shape(evals) != np.shape(occs):
            raise ValueError("evals and occs should have same shape.")
        try:
            return float(evals[occs > tol].max())
        except:
            raise ValueError("Fermi energy/VBM could not be determined form given evals and occs.")
    
    # Following methods should be implemented in a subclass
    def get_summary(self): raise NotImplementedError("`get_summary` should be implemented in a subclass. See Vasprun.get_summary as example.")
    
    def get_structure(self): raise NotImplementedError("`get_structure` should be implemented in a subclass. See Vasprun.get_structure as example.")
    
    def get_kpoints(self): raise NotImplementedError("`get_kpoints` should be implemented in a subclass. See Vasprun.get_kpoints as example.")
    
    def get_evals(self, *args, **kwargs): raise NotImplementedError("`get_evals` should be implemented in a subclass. See Vasprun.get_evals as example.")
    
    def get_dos(self, *args, **kwargs): raise NotImplementedError("`get_dos` should be implemented in a subclass. See Vasprun.get_dos as example.")
    
    def get_spins(self, *args, **kwargs): raise NotImplementedError("`get_spins` should be implemented in a subclass. See Vasprun.get_spins as example.")
    
    def get_forces(self): raise NotImplementedError("`get_forces` should be implemented in a subclass. See Vasprun.get_forces as example.")
    
    def get_scsteps(self): raise NotImplementedError("`get_scsteps` should be implemented in a subclass. See Vasprun.get_scsteps as example.")
    

class Vasprun(DataSource):
    "Reads vasprun.xml file lazily. It reads only the required data from the file when a plot or data access is requested."
    def __init__(self, path = './vasprun.xml', skipk = None):
        super().__init__(path)
        self._skipk = skipk if isinstance(skipk,int) else self.get_skipk()
        
    def read(self, start_match, stop_match, nth_match = 1, skip_last = False):
        """Reads a part of the file between start_match and stop_match and returns a generator. It is lazy and fast.
        `start_match` and `stop_match` are regular expressions. `nth_match` is the number of occurence of start_match to start reading.
        `skip_last` is used to determine whether to keep or skip last line.
        """
        if '|' in start_match:
            raise ValueError("start_match should be a single match, so '|' character is not allowed.")
        
        with open(self.path) as f:
            lines = islice(f,None) # this is fast
            matched = False
            n_start = 1
            for line in lines:
                if re.search(start_match, line, flags = re.DOTALL):
                    if nth_match != n_start:
                        n_start +=1
                    else:
                        matched = True
                if matched and re.search(stop_match,line, flags = re.DOTALL): # avoid stop before start
                    matched = False
                    if not skip_last:
                        yield line
                        
                    break # stop reading
                if matched: # should be after break to handle last line above
                    yield line
                
    def get_skipk(self):
        "Returns the number of k-points to skip in band structure plot in case of HSE calculation."
        weights = np.fromiter(
            (v.text for v in ET.fromstringlist(self.read('<varray.*weights','</varray')).iter('v')),
            dtype = float)
        return len([w for w in weights if w != weights[-1]])
    
    def get_summary(self):
        "Returns summary data of calculation."
        incar = {v.attrib['name']: v.text for v in ET.fromstringlist(self.read('<incar','</incar')).iter('i')}
        
        info_dict = {'SYSTEM':incar['SYSTEM']}
        info_dict['ISPIN']   = int(ET.fromstring(next(self.read('<i.*ISPIN','</i>'))).text)
        info_dict['NBANDS']  = int(ET.fromstring(next(self.read('<i.*NBANDS','</i>'))).text)
        info_dict['NELECTS'] = int(float(ET.fromstring(next(self.read('<i.*NELECT','</i>'))).text)) # Beacuse they are poorly writtern as float, come on VASP!
        info_dict['LSORBIT'] = True if 't' in ET.fromstring(next(self.read('<i.*LSORBIT','</i>'))).text.lower() else False
        info_dict['EFERMI']  = float(ET.fromstring(next(self.read('<i.*efermi','</i>'))).text)
        
        # Getting ions is kind of terrbile, but it works
        atominfo = ET.fromstringlist(self.read('<atominfo','</atominfo'))
        info_dict['NIONS'] = [int(atom.text) for atom in atominfo.iter('atoms')][0]
        
        types  = [int(_type.text) for _type in atominfo.iter('types')][0]
        elems  = [rc[0].text.strip() for rc in atominfo.iter('rc')]
        _inds  = [int(a) for a in elems[-types:]]

        inds = np.cumsum([0,*_inds]).astype(int)
        names = list(np.unique(elems[:-types]))
        info_dict['types'] = {name:range(inds[i],inds[i+1]) for i,name in enumerate(names)}
    
        # Getting projection fields
        fields = [f.replace('field','').strip(' ></\n') for f in self.read('<partial','<set') if 'field' in f] # poor formatting again
        if fields:
            info_dict['orbs'] = [f for f in fields if 'energy' not in f] # projection fields
        
        info_dict['incar'] = incar # Just at end 
        return serializer.Dict2Data(info_dict)
    
    def get_structure(self):
        "Returns a structure object including types, basis, rec_basis and positions."
        arrays = np.array([[float(i) for i in v.split()[1:4]] for v in self.read('<structure.*finalpos','select|</structure') if '<v>' in v]) # Stop at selctive dynamics if exists
        info = self.get_summary()
        return serializer.PoscarData({
            
            
            'SYSTEM':info.SYSTEM,
            'basis': arrays[:3],
            'rec_basis':arrays[3:6],
            'positions':arrays[6:],
            'types':info.types,
            'extra_info': {'comment':'Exported from vasprun.xml','cartesian':False,'scale':1}
            })
        
    def get_kpoints(self):
        "Returns k-points data including kpoints, coords, weights and rec_basis in which coords are calculated."
        kpoints = np.array([[float(i) for i in v.text.split()] for v in ET.fromstringlist(self.read('<varray.*kpoint','</varray')).iter('v')])[self._skipk:]
        weights = np.fromiter((v.text for v in ET.fromstringlist(self.read('<varray.*weights','</varray')).iter('v')),dtype = float)[self._skipk:]
        # Get rec_basis to make cartesian coordinates
        rec_basis = np.array([[float(i) for i in v.split()[1:4]] for v in self.read('<structure.*finalpos','select|</structure') if '<v>' in v])[3:6] # could be selective dynamics there
        coords = kpoints.dot(rec_basis) # cartesian coordinates
        kpath = np.cumsum([0, *np.linalg.norm(coords[1:] - coords[:-1],axis=1)])
        kpath = kpath/kpath[-1] # normalized kpath to see the band structure of all materials in same scale
        return serializer.Dict2Data({'kpoints':kpoints,'coords':coords,'kpath': kpath, 'weights':weights,'rec_basis':rec_basis})
     
     
    def get_dos(self, spin = 0, elim = None, atoms = None):
        # Should be as energy(NE,), total(NE,), integrated(NE,), partial(NE, NATOMS(selected), NORBS) and atoms reference should be returned
        pass 
    
    def _get_spin_set(self, spin, bands_range, atoms): # bands_range comes from elim applied to all bands
        if atoms == -1:
            pass 
        elif isinstance(atoms, (list, tuple, range)):
            pass 
        else:
            raise ValueError('atoms should be a list, tuple, range or -1 for all atoms')
        return None
    
    def get_evals(self, spin = 0, elim = None, atoms = None): 
        # mention that elim applies to unsubstracted efermi from evals, and elim should keep track of efermi, inside Bands and DOS, add efermi to elim passed to get_evals from bands
        info = self.get_summary()
        bands_range = range(info.NBANDS)
        
        ev = (r.split()[1:3] for r in self.read(f'<set.*spin {spin+1}',f'<set.*spin {spin+2}|</eigenvalues>') if '<r>' in r)
        ev = (e for es in ev for e in es) # flatten
        ev = np.fromiter(ev,float)
        if ev.size:
            ev = ev.reshape((-1,info.NBANDS,2))[self._skipk:]
        else:
            raise ValueError('No eigenvalues found for spin {}'.format(spin))
        evals, occs = ev[:,:,0], ev[:,:,1]
        
        if elim:
            up_ind = np.max(np.where(evals[:,:] <= np.max(elim))[1]) + 1
            lo_ind = np.min(np.where(evals[:,:] >= np.min(elim))[1])
            evals = evals[:, lo_ind:up_ind]
            occs = occs[:, lo_ind:up_ind]
            bands_range = range(lo_ind, up_ind)
        
        out = {'evals':evals,'occs':occs}
        if atoms:
            # Get projections here
            out['pros'] = self._get_spin_set(spin, bands_range, atoms)
            out['atoms'] = atoms
            
        return serializer.Dict2Data(out)

    def get_spins(self, bands = -1, atoms = -1): 
        # Just have a loop over _get_spin_set and collect all spins, evals must be collected for both spin up and down
        # shape should be (ISPIN, NKPOINTS, NBANDS, [NATOMS(selected), NORBS]) and atoms and bands reference should be returned
        pass
    
    def get_forces(self):
        "Reads force on each ion from vasprun.xml"
        node = ET.fromstringlist(self.read('<varray.*forces','</varray>'))
        return np.array([[float(i) for i in v.text.split()] for v in node.iter('v')])
    
    def get_scsteps(self):
        "Reads all self-consistent steps from vasprun.xml"
        node = ET.fromstringlist(['<steps>', *self.read('<.*scstep','<structure>', skip_last = True),'</steps>']) # poor formatting 
        steps = []
        for e in node.iter('energy'):
            _d = {}
            for i in e.iter('i'):
                if '_en' in i.attrib['name']:
                    _d[i.attrib['name']] = float(i.text)
            steps.append(_d)
            
        if steps:
            arrays = {k:[] for k in steps[0].keys()}
            for step in steps:
                for k,v in step.items():
                    arrays[k].append(v)

            return serializer.Dict2Data({k:np.array(v) for k,v in arrays.items()})
        

def xml2dict(xmlnode_or_filepath):
    """Convert xml node or xml file content to dictionary. All output text is in string format, so further processing is required to convert into data types/split etc.
    
    Args:
        xmlnode_or_filepath: It is either a path to an xml file or an ``xml.etree.ElementTree.Element`` object.
        
    Each node has ``tag,text,attr,nodes`` attributes. Every text element can be accessed via
    ``xml2dict()['nodes'][index]['nodes'][index]...`` tree which makes it simple.
    """
    if isinstance(xmlnode_or_filepath,str):
        node = read_asxml(xmlnode_or_filepath)
    else:
        node = xmlnode_or_filepath

    text = node.text.strip() if node.text else ''
    nodes = [xml2dict(child) for child in list(node)]
    return {'tag': node.tag,'text': text, 'attr':node.attrib, 'nodes': nodes}


def get_tdos(xml_data,elim = []):
    tdos=[]; #assign for safely exit if wrong spin set entered.
    ISPIN = get_ispin(xml_data=xml_data)
    for neighbor in xml_data.root.iter('dos'):
        for item in neighbor[1].iter('set'):
            if ISPIN == 1:
                if item.attrib == {'comment': 'spin 1'}:
                    tdos = np.array([[[float(entry) for entry in arr.text.split()] for arr in item]])
            if ISPIN == 2:
                if item.attrib == {'comment': 'spin 1'}:
                    tdos_1 = [[float(entry) for entry in arr.text.split()] for arr in item]
                if item.attrib=={'comment': 'spin 2'}:
                    tdos_2 = [[float(entry) for entry in arr.text.split()] for arr in item]
                tdos = np.array([tdos_1,tdos_2])
            
    for i in xml_data.root.iter('i'): #efermi for condition required.
        if i.attrib == {'name': 'efermi'}:
            efermi = float(i.text)
    dos_dic= {'Fermi':efermi,'ISPIN':ISPIN,'tdos':tdos}
    #Filtering in energy range.
    if elim: #check if elim not empty
        up_ind = np.max(np.where(tdos[:, :, 0] - efermi <= np.max(elim))[1]) + 1
        lo_ind = np.min(np.where(tdos[:, :, 0] - efermi >= np.min(elim))[1])
        tdos = tdos[:,lo_ind:up_ind,:]
        dos_dic= {'Fermi':efermi,'ISPIN':ISPIN,'grid_range':range(lo_ind,up_ind),'tdos':tdos}
    return serializer.Dict2Data(dos_dic)

def get_evals(xml_data, skipk = None, elim = []):
    evals, occs = [], [] #assign for safely exit if wrong spin set entered.
    ISPIN = get_ispin(xml_data=xml_data)
    if skipk != None:
        skipk=skipk
    else:
        skipk = exclude_kpts(xml_data=xml_data) #that much to skip by default
    for neighbor in xml_data.root.iter('eigenvalues'):
        for item in neighbor[0].iter('set'):
            if ISPIN == 1:
                if item.attrib=={'comment': 'spin 1'}:
                    evals = np.array([[[float(t) for t in th.text.split()] for th in thing] for thing in item])[skipk:]
                    evals, occs = np.expand_dims(evals[:,:,0],0), np.expand_dims(evals[:,:,1],0)
                    NBANDS = len(evals[0])
            if ISPIN == 2:
                if item.attrib=={'comment': 'spin 1'}:
                    eval_1 = np.array([[[float(t) for t in th.text.split()] for th in thing] for thing in item])[skipk:]
                    eval_1, occs_1 = eval_1[:,:,0], eval_1[:,:,1]
                if item.attrib=={'comment': 'spin 2'}:
                    eval_2=np.array([[[float(t) for t in th.text.split()] for th in thing] for thing in item])[skipk:]
                    eval_2, occs_2 = eval_2[:,:,0], eval_2[:,:,1]
                
                evals = np.array([eval_1,eval_2])
                occs = np.array([occs_1,occs_2])
                NBANDS = evals.shape[-1]

    for i in xml_data.root.iter('i'): #efermi for condition required.
        if i.attrib == {'name': 'efermi'}:
            efermi = float(i.text)
    evals_dic={'Fermi':efermi,'ISPIN':ISPIN,'NBANDS':NBANDS,'evals':evals,'occs':occs}
    if elim: #check if elim not empty
        up_ind = np.max(np.where(evals[:,:,:]-efermi <= np.max(elim))[2]) + 1
        lo_ind = np.min(np.where(evals[:,:,:]-efermi >= np.min(elim))[2])
        evals = evals[:, :, lo_ind:up_ind]
        occs = occs[:, :, lo_ind:up_ind]
        
        NBANDS = int(up_ind - lo_ind) #update Bands
        evals_dic['NBANDS'] = NBANDS
        evals_dic['evals'] = evals
        evals_dic['occs'] = occs
        
    return serializer.Dict2Data(evals_dic)

def get_bands_pro_set(xml_data, spin = 0, skipk = 0, bands_range:range=None, set_path:str=None):
    """Returns bands projection of a spin(default 0). If spin-polarized calculations, gives SpinUp and SpinDown keys as well.
    
    Args:
        xml_data    : From ``read_asxml`` function's output.
        skipk (int): Number of initil kpoints to skip (Default 0).
        spin (int): Spin set to get, default is 0.
        bands_range (range): If elim used in ``get_evals``,that will return ``bands_range`` to use here. Note that range(0,2) will give 2 bands 0,1 but tuple (0,2) will give 3 bands 0,1,2.
        set_path (str): path/to/_set[1,2,3,4].txt, works if ``split_vasprun`` is used before.
    
    Returns:
        Dict2Data: ``ipyvasp.Dict2Data`` with attibutes of bands projections and related parameters.
    """
    if bands_range != None:
        check_list = list(bands_range)
        if check_list==[]:
            raise ValueError("No bands prjections found in given energy range.")
    # Try to read _set.txt first. instance check is important.
    if isinstance(set_path,str) and os.path.isfile(set_path):
        _header = islice2array(set_path,nlines=1,raw=True,exclude=None)
        _shape = [int(v) for v in _header.split('=')[1].strip().split(',')]
        NKPTS, NBANDS, NIONS, NORBS = _shape
        if NORBS == 3:
            fields = ['s','p','d']
        elif NORBS == 9:
            fields = ['s','py','pz','px','dxy','dyz','dz2','dxz','x2-y2']
        else:
            fields = [str(i) for i in range(NORBS)] #s,p,d in indices.
        COUNT = NIONS*NBANDS*(NKPTS-skipk)*NORBS
        start = NBANDS*NIONS*skipk
        nlines = None # Read till end.
        if bands_range:
            _b_r = list(bands_range)
            # First line is comment but it is taken out by exclude in islice2array.
            start = [[NIONS*NBANDS*k + NIONS*b for b in _b_r] for k in range(skipk,NKPTS)]
            start = [s for ss in start for s in ss] #flatten
            nlines = NIONS # 1 band has nions
            NBANDS = _b_r[-1]-_b_r[0]+1 # upadte after start

        NKPTS = NKPTS-skipk # Update after start, and bands_range.
        COUNT = NIONS*NBANDS*NKPTS*NORBS
        data = islice2array(set_path,start=start,nlines=nlines,count=COUNT)
        data = data.reshape((1, NKPTS,NBANDS,NIONS,NORBS))  # 1 for spin to just be consistent
        return serializer.Dict2Data({'labels':fields,'pros':data})

    #Collect Projection fields
    fields=[];
    for pro in xml_data.root.iter('projected'):
        for arr in pro.iter('field'):
            if('eig' not in arr.text and 'occ' not in arr.text):
                fields.append(arr.text.strip())
    NORBS = len(fields)
    #Get NIONS for reshaping data
    NIONS=[int(atom.text) for atom in xml_data.root.iter('atoms')][0]

    for sp in xml_data.root.iter('set'):
        if sp.attrib=={'comment': 'spin{}'.format(spin + 1)}:
            k_sets = [kp for kp in sp.iter('set') if 'kpoint' in kp.attrib['comment']]
    k_sets = k_sets[skipk:]
    NKPTS = len(k_sets)
    band_sets = []
    for k_s in k_sets:
        b_set = [b for b in k_s.iter('set') if 'band' in b.attrib['comment']]
        if bands_range == None:
            band_sets.extend(b_set)
        else:
            b_r = list(bands_range)
            band_sets.extend(b_set[b_r[0]:b_r[-1]+1])
    NBANDS = int(len(band_sets)/len(k_sets))
    try:
        # Error prone solution but 5 times fater than list comprehension.
        bands_pro = (float(t) for band in band_sets for l in band.iter('r') for t in l.text.split())
        COUNT = NKPTS*NBANDS*NORBS*NIONS # Must be counted for performance.
        data = np.fromiter(bands_pro,dtype=float,count=COUNT)
    except:
        # Alternate slow solution
        print("Error using `np.fromiter`.\nFalling back to (slow) list comprehension...",end=' ')
        bands_pro = (l.text for band in band_sets for l in band.iter('r'))
        bands_pro = [[float(t) for t in text.split()] for text in bands_pro]
        data = np.array(bands_pro)
        del bands_pro # Release memory
        print("Done.")

    data = data.reshape((1, NKPTS,NBANDS,NIONS,NORBS)) # extra dim for spin
    return serializer.Dict2Data({'labels':fields,'pros':data})

def get_dos_pro_set(xml_data,spin = 0,dos_range:range=None):
    """Returns dos projection of a spin(default 0) as numpy array. If spin-polarized calculations, gives SpinUp and SpinDown keys as well.
    
    Args:
        xml_data : From ``read_asxml`` function
        spin (int): Spin set to get, default 0.
        dos_range (range): If elim used in ``get_tdos``,that will return dos_range to use here..
    
    Returns:
        Dict2Data : ``ipyvasp.Dict2Data`` with attibutes of dos projections and related parameters.
    """
    if dos_range != None:
        check_list = list(dos_range)
        if check_list == []:
            raise ValueError("No DOS prjections found in given energy range.")

    n_ions=get_summary(xml_data=xml_data).NION
    for pro in xml_data.root.iter('partial'):
        dos_fields=[field.text.strip()for field in pro.iter('field')]
        #Collecting projections.
        dos_pro=[]; set_pro=[]; #set_pro=[] in case spin set does not exists
        for ion in range(n_ions):
            for node in pro.iter('set'):
                if(node.attrib=={'comment': 'ion {}'.format(ion+1)}):
                    for sp in node.iter('set'):
                        if(sp.attrib=={'comment': 'spin {}'.format(spin + 1)}):
                            set_pro=[[float(entry) for entry in r.text.split()] for r in sp.iter('r')]
            dos_pro.append(set_pro)
    if dos_range==None: #full grid computed.
        dos_pro=np.array(dos_pro) # shape(NION,e_grid,pro_fields)
    else:
        dos_range=list(dos_range)
        min_ind=dos_range[0]
        max_ind=dos_range[-1]+1
        dos_pro=np.array(dos_pro)[:,min_ind:max_ind,:]
    final_data = np.expand_dims(dos_pro,0).transpose((0,2,1,3)) # shape(1, NE,NIONS, NORBS + 1)
    return serializer.Dict2Data({'labels':dos_fields,'pros':final_data})



def export_vasprun(path:str = None, skipk:int = None, elim:list = [], dos_only:bool = False):
    """Returns a full dictionary of all objects from ``vasprun.xml`` file. It first try to load the data exported by powershell's `Export-VR(Vasprun)`, which is very fast for large files. It is recommended to export large files in powershell first.
    
    Args:
        path  (str): Path to ``vasprun.xml`` file. Default is ``'./vasprun.xml'``.
        skipk (int): Default is None. Automatically detects kpoints to skip.
        elim  (list): List [min,max] of energy interval. Default is [], covers all bands.
        dos_only (bool): If True, only returns dos data with minimal other data. Default is False.
        
    Returns:
        VasprunData: ``ipyvasp.serializer.VasprunData`` object.
    """
    path = path or './vasprun.xml'

    xml_data = read_asxml(path=path)

    base_dir = os.path.split(os.path.abspath(path))[0]
    set_paths = [os.path.join(base_dir,"_set{}.txt".format(i)) for i in (1,2)]
    #First exclude unnecessary kpoints. Includes only same weight points
    if skipk!=None:
        skipk=skipk
    else:
        skipk = exclude_kpts(xml_data) #that much to skip by default
    info_dic = get_summary(xml_data) #Reads important information of system.
    #KPOINTS
    kpts = get_kpoints(xml_data,skipk=skipk)
    #EIGENVALS
    eigenvals = get_evals(xml_data,skipk=skipk,elim=elim)
    #TDOS
    tot_dos = get_tdos(xml_data,elim = elim)
    #Bands and DOS Projection
    if elim:
        bands_range = eigenvals.indices #indices in range form.
        grid_range=tot_dos.grid_range
    else:
        bands_range = None #projection function will read itself.
        grid_range = None
        
    if dos_only:
        bands_range = range(1) # Just one band
        skipk = len(kpts.kpath) + skipk - 2 # Just Single kpoint
        
    if info_dic.ISPIN == 1:
        pro_bands = get_bands_pro_set(xml_data=xml_data,spin = 0,skipk=skipk,bands_range=bands_range,set_path=set_paths[0])
        pro_dos = get_dos_pro_set(xml_data=xml_data,spin = 0,dos_range=grid_range)
    if info_dic.ISPIN == 2:
        pro_1 = get_bands_pro_set(xml_data=xml_data,spin =0,skipk=skipk,bands_range=bands_range,set_path=set_paths[0])
        pro_2 = get_bands_pro_set(xml_data=xml_data,spin =1,skipk=skipk,bands_range=bands_range,set_path=set_paths[1])
        pros = np.vstack([pro_1.pros,pro_2.pros]) # accessing spins in dictionary after .pro.
        pro_bands = {'labels':pro_1.labels,'pros': pros}
        pdos_1 = get_dos_pro_set(xml_data=xml_data,spin =0,dos_range=grid_range)
        pdos_2 = get_dos_pro_set(xml_data=xml_data,spin=0,dos_range=grid_range)
        pdos = np.vstack([pdos_1.pros, pdos_2.pros]) # accessing spins in dictionary after .pro.
        pro_dos = {'labels':pdos_1.labels,'pros': pdos}
        
    # Forces and steps
    force = get_force(xml_data)
    scsteps = get_scsteps(xml_data)

    #Structure
    poscar = get_structure(xml_data = xml_data)
    poscar = {'SYSTEM':info_dic.SYSTEM,**poscar.to_dict()}
    #Dimensions dictionary.
    dim_dic={'kpoints':'(NKPTS,3)','kpath':'(NKPTS,1)','evals':'⇅(NSPIN,NKPTS,NBANDS)','dos_total':'⇅(NSPIN, NE,3)','dos_partial':'⇅(NSPIN, NE, NIONS,NORBS+1)','evals_projector':'⇅(NSPIN,NKPTS,NBANDS,NIONS, NORBS)'}
    # Bands
    bands = {'kpoints':kpts.kpoints,'kpath':kpts.kpath,'evals':eigenvals.evals,'occs':eigenvals.occs, 'labels':pro_bands['labels'],'pros':pro_bands['pros']}
    # DOS
    dos = {'total':tot_dos.tdos,'labels':pro_dos['labels'],'partial':pro_dos['pros']}
    #Writing everything to be accessible via dot notation
    full_dic={'sys_info':info_dic,'dim_info':dim_dic,'bands':bands,'dos':dos,'poscar': poscar,'force':force,'scsteps':scsteps}
    return serializer.VasprunData(full_dic)

def _validate_evr(path_evr=None,**kwargs):
    "Validates data given for plotting functions. Returns a tuple of (Boolean,data)."
    if type(path_evr) == serializer.VasprunData:
        return path_evr

    path_evr = path_evr or './vasprun.xml' # default path.

    if isinstance(path_evr,str):
        if os.path.isfile(path_evr):
            # kwargs -> skipk=skipk,elim=elim
            return export_vasprun(path=path_evr,**kwargs)
        else:
            raise FileNotFoundError(f'File {path_evr!r} not found!')
    # Other things are not valid.
    raise ValueError('path_evr must be a path string or output of export_vasprun function.')

def islice2array(path_or_islice,dtype = float,delimiter:str = '\s+',
                include:str=None,exclude:str='#',raw:bool=False,fix_format:bool = True,
                start:int=0,nlines:int=None,count:int=-1,cols: tuple =None,new_shape : tuple=None
                ):
    """Reads a sliced array from txt,csv type files and return to array. Also manages if columns lengths are not equal and return 1D array. 
    It is faster than loading  whole file into memory. This single function could be used to parse EIGENVAL, PROCAR, DOCAR and similar files 
    with just a combination of ``exclude, include,start,stop,step`` arguments. See code of ``ipyvasp.parser.export_locpot`` for example.
    
    Args:
        path_or_islice: Path/to/file or ``itertools.islice(file_object)``. islice is interesting when you want to read different slices of 
            an opened file and do not want to open it again. 
        dtype (conversion function): float by default. Data type of output array, it is must have argument.
        start (int): Starting line number. Default is 0. It could be a list to read slices from file provided that nlines is int. 
            The spacing between adjacent indices in start should be equal to or greater than nlines as pointer in file do not go back on its own.
            ``start`` should count comments if ``exclude`` is None. You can use ``slice_data`` function to get a dictionary of 
            ``start,nlines, count, cols, new_shape`` and unpack in argument instead of thinking too much.
        nlines (int): Number of lines after start respectively. Only work if ``path_or_islice`` is a file path. could be None or int.
        count (int): Default is -1. ``count = np.size(output_array) = nrows x ncols``, if it is known before execution, performance is increased. This parameter is in output of ``slice_data``.
        delimiter (str):  Default is `\s+`. Could be any kind of delimiter valid in numpy and in the file.
        cols (tuple): Tuple of indices of columns to pick. Useful when reading a file like PROCAR which e.g. has text and numbers inline. This parameter is in output of ``slice_data``.
        include (str): Default is None and includes everything. String of patterns separated by | to keep, could be a regular expression.
        exclude (str): Default is '#' to remove comments. String of patterns separated by | to drop,could be a regular expression.
        raw (bool): Default is False, if True, returns list of raw strings. Useful to select ``cols``.
        fix_format (bool): Default is True, it sepearates numbers with poor formatting like ``1.000-2.000`` to ``1.000 -2.000`` which is useful in PROCAR. Keep it False if want to read string literally.
        new_shape (tuple): Tuple of shape Default is None. Will try to reshape in this shape, if fails fallbacks to 2D or 1D. This parameter is in output of ``slice_data``.
    
    Returns:
        1D or 2D array of dtype. If raw is True, returns raw data.
        
    .. code-block:: python
        :caption: **Example**
        
        islice2array('path/to/PROCAR',start=3,include='k-point',cols=[3,4,5])[:2]
        array([[ 0.125,  0.125,  0.125],
               [ 0.375,  0.125,  0.125]])
        islice2array('path/to/EIGENVAL',start=7,exclude='E',cols=[1,2])[:2]
        array([[-11.476913,   1.      ],
               [  0.283532,   1.      ]])
    
    .. note::
        Slicing a dimension to 100% of its data is faster than let say 80% for inner dimensions, so if you have to slice more than 50% of an inner dimension, then just load full data and slice after it.
    """
    if nlines is None and isinstance(start,(list,np.ndarray)):
        print("`nlines = None` with `start = array/list` is useless combination.")
        return np.array([]) # return empty array.

    def _fixing(_islice,include=include, exclude=exclude,fix_format=fix_format,nlines=nlines,start=start):
        if include:
            _islice = (l for l in _islice if re.search(include,l))

        if exclude:
            _islice = (l for l in _islice if not re.search(exclude,l))

        _islice = (l.strip() for l in _islice) # remove whitespace and new lines

        # Make slices here after comment excluding.
        if isinstance(nlines,int) and isinstance(start,(list,np.ndarray)):
            #As islice moves the pointer as it reads, start[1:]-nlines-1
            # This confirms spacing between two indices in start >= nlines
            start = [start[0],*[s2-s1-nlines for s1,s2 in zip(start,start[1:])]]
            _islice = chain(*(islice(_islice,s,s+nlines) for s in start))
        elif isinstance(nlines,int) and isinstance(start,int):
            _islice = islice(_islice,start,start+nlines)
        elif nlines is None and isinstance(start,int):
            _islice = islice(_islice,start,None)

        # Negative connected digits to avoid, especially in PROCAR
        if fix_format:
            _islice = (re.sub(r"(\d)-(\d)",r"\1 -\2",l) for l in _islice)
        return _islice

    def _gen(_islice,cols=cols):
        for line in _islice:
            line = line.strip().replace(delimiter,'  ').split()
            if line and cols is not None: # if is must here.
                line = [line[i] for i in cols]
            for chars in line:
                yield dtype(chars)

    #Process Now
    if isinstance(path_or_islice,str) and os.path.isfile(path_or_islice):
        with open(path_or_islice,'r') as f:
            _islice = islice(f,0,None) # Read full, Will fix later.
            _islice = _fixing(_islice)
            if raw:
                return '\n'.join(_islice)
            # Must to consume islice when file is open
            data = np.fromiter(_gen(_islice),dtype=dtype,count=count)
    else:
        _islice = _fixing(path_or_islice)
        if raw:
            return '\n'.join(_islice)
        data = np.fromiter(_gen(_islice),dtype=dtype,count=count)

    if new_shape:
        try: data = data.reshape(new_shape)
        except: pass
    elif cols: #Otherwise single array.
        try: data = data.reshape((-1,len(cols)))
        except: pass
    return data

def slice_data(dim_inds:list,old_shape:tuple):
    """Returns a dictionary that can be unpacked in arguments of isclice2array function. This function works only for regular txt/csv/tsv data files which have rectangular data written.
    
    Args:
        dim_inds (list): List of indices array or range to pick from each dimension. Inner dimensions are more towards right. Last itmes in dim_inds is considered to be columns. If you want to include all values in a dimension, you can put -1 in that dimension. Note that negative indexing does not work in file readig, -1 is s special case to fetch all items.
        old_shape (tuple): Shape of data set including the columns length in right most place.
    
    Suppose You have data as 3D arry where third dimension is along column.
    
    .. code-block:: shell
        :caption: data.txt
        
        0 0
        0 2
        1 0
        1 2
        
    To pick [[0,2], [1,2]], i.e. second and fourth row, you need to run
    
    .. code-block:: python
        :caption: slicing data
        
        slice_data(dim_inds = [[0,1],[1],-1], old_shape=(2,2,2))
        {'start': array([1, 3]), 'nlines': 1, 'count': 2}
        
    Unpack above dictionary in ``islice2array`` and you will get output array.
    
    .. note::
        The dimensions are packed from right to left, like 0,2 is repeating in 2nd column.
    """
    # Columns are treated diffiernetly.
    if dim_inds[-1] == -1:
        cols = None
    else:
        cols = list(dim_inds[-1])

    r_shape = old_shape[:-1]
    dim_inds = dim_inds[:-1]
    for i,ind in enumerate(dim_inds.copy()):
        if ind == -1:
            dim_inds[i] = range(r_shape[i])
    nlines = 1
    #start = [[NIONS*NBANDS*k + NIONS*b for b in _b_r] for k in range(skipk,NKPTS)] #kind of thing.
    _prod_ = product(*dim_inds)
    _mult_ = [np.product(r_shape[i+1:]) for i in range(len(r_shape))]
    _out_ = np.array([np.dot(p,_mult_) for p in _prod_]).astype(int)
    # check if innermost dimensions could be chunked.
    step = 1
    for i in range(-1,-len(dim_inds),-1):
        _inds = np.array(dim_inds[i]) #innermost
        if np.max(_inds[1:] - _inds[:-1]) == 1: # consecutive
            step = len(_inds)
            _out_ = _out_[::step] # Pick first indices
            nlines = step*nlines
            # Now check if all indices picked then make chunks in outer dimensions too.
            if step != r_shape[i]: # Can't make chunk of outer dimension if inner is not 100% picked.
                break # Stop more chunking
    new_shape = [len(inds) for inds in dim_inds] #dim_inds are only in rows.
    new_shape.append(old_shape[-1])
    return {'start':_out_,'nlines':nlines,'count': nlines*len(_out_),'cols':cols,'new_shape':tuple(new_shape)}

def split_vasprun(path:str = None):
    """Splits a given vasprun.xml file into a smaller _vasprun.xml file plus _set[1,2,3,4].txt files which contain projected data for each spin set.
    
    Args:
        path (str): path/to/vasprun.xml file.
    
    Output:
        - _vasprun.xml file with projected data.
        - _set1.txt for projected data of colinear calculation.
        - _set1.txt for spin up data and _set2.txt for spin-polarized case.
        - _set[1,2,3,4].txt for each spin set of non-colinear calculations.
    """
    if not path:
        path = './vasprun.xml'
    if not os.path.isfile(path):
        raise FileNotFoundError("{!r} does not exist!".format(path))
    base_dir = os.path.split(os.path.abspath(path))[0]
    out_file = os.path.join(base_dir,'_vasprun.xml')
    out_sets = [os.path.join(base_dir,'_set{}.txt'.format(i)) for i in range(1,5)]
    # process
    with open(path,'r') as f:
        lines = islice(f,None)
        indices = [i for i,l in enumerate(lines) if re.search('projected|/eigenvalues',l)]
        f.seek(0)
        print("Writing {!r} ...".format(out_file),end=' ')
        with open(out_file,'w') as outf:
            outf.write(''.join(islice(f,0,indices[1])))
            f.seek(0)
            outf.write(''.join(islice(f,indices[-1]+1,None)))
            print('Done')

        f.seek(0)
        middle = islice(f,indices[-2]+1,indices[-1]) #projected words excluded
        spin_inds = [i for i,l in enumerate(middle) if re.search('spin',l)][1:] #first useless.
        if len(spin_inds)>1:
            set_length = spin_inds[1]-spin_inds[0] # Must define
        else:
            set_length = indices[-1]-indices[-2] #It is technically more than set length, but fine for 1 set
        f.seek(0) # Must be at zero
        N_sets = len(spin_inds)
        # Let's read shape from out_file as well.
        xml_data = read_asxml(out_file)
        _summary = get_summary(xml_data)
        NIONS  = _summary.NION
        NORBS  = len(_summary.fields)
        NBANDS = get_evals(xml_data).NBANDS
        NKPTS  = get_kpoints(xml_data).NKPTS
        del xml_data # free meory now.
        for i in range(N_sets): #Reads every set
            print("Writing {!r} ...".format(out_sets[i]),end=' ')
            start = (indices[-2]+1+spin_inds[0] if i==0 else 0) # pointer is there next time.
            stop_ = start + set_length # Should move up to set length only.
            with open(out_sets[i],'w') as setf:
                setf.write("  # Set: {} Shape: (NKPTS[NBANDS[NIONS]],NORBS) = {},{},{},{}\n".format(i+1,NKPTS,NBANDS,NIONS,NORBS))
                middle = islice(f,start,stop_)
                setf.write(''.join(l.lstrip().replace('/','').replace('<r>','') for l in middle if '</r>' in l))
                print('Done')

def export_spin_data(path = None, spins = 's', skipk = None, elim = None):
    """Returns Data with selected spin sets. For spin polarized calculations, it returns spin up and down data.
    
    Args:
        path  (str): Path to ``vasprun.xml`` file. Default is `'./vasprun.xml'`.
        skipk (int): Default is None. Automatically detects kpoints to skip.
        elim  (list): List [min,max] of energy interval. Default is [], covers all bands.
        spins (str): Spin components to include from 'sxyz', e.g. 'sx' will pick <S> and <S_x> if present.
        Only works if ISPIN == 1, otherwise it will be two sets for spin up and down.
    
    Returns:
        SpinData: ``ipyvasp.serializer.SpinData`` object.
    """
    if not isinstance(spins,str):
        raise TypeError(f"`spins` must be a string from 'sxyz', got {spins}!")

    if False in [comp in 'sxyz' for comp in spins]:
        raise ValueError(f"`spins` must be in 'sxyz', got {spins!r}!")

    xml_data = read_asxml(path = path or './vasprun.xml')

    base_dir = os.path.split(os.path.abspath(path or './vasprun.xml'))[0]
    set_paths = [os.path.join(base_dir,"_set{}.txt".format(i)) for i in (1,2,3,4)]

    skipk = skipk or exclude_kpts(xml_data=xml_data) #that much to skip by default
    full_dic = {'sys_info':get_summary(xml_data)}

    ISPIN = full_dic['sys_info'].ISPIN
    LSORBIT = getattr(full_dic['sys_info'].incar, 'LSORBIT', 'FALSE')
    if 'f' in LSORBIT.lower() and ISPIN == 1:
        for comp in spins:
            if comp in 'xyz':
                raise ValueError(f"LSORBIT = {LSORBIT} does not include spin component {comp!r}!")

    full_dic['dim_info'] = {'kpoints':'(NKPTS,3)','evals.<e,u,d>':'⇅(NKPTS,NBANDS)','spins.<u,d,s,x,y,z>':'⇅(NION,NKPTS,NBANDS,pro_fields)'}
    full_dic['kpoints']= get_kpoints(xml_data, skipk = skipk).kpoints

    bands = get_evals(xml_data, skipk = skipk,elim = elim).to_dict()
    evals = bands['evals']
    bands.update({'u': evals[0], 'd': evals[1]} if ISPIN == 2 else {'e': evals[0]})
    
    del bands['evals'] # Do not Delete occupancies here
    full_dic['evals'] = bands

    bands_range = full_dic['bands'].indices if elim else None #indices in range form.

    spin_sets = {}
    if ISPIN == 1:
        for n, s in enumerate('sxyz', start = 0): # spins 0,1,2,3 for s,x,y,z
            if s in spins:
                spin_sets[s] = get_bands_pro_set(xml_data, spin = n, skipk = skipk, bands_range = bands_range, set_path = set_paths[n-1]).pros[0] # remove extra dimension

    if ISPIN == 2:
        print(utils.color.g(f"Found ISPIN = 2, output data got attributes spins.<u,d> instead of spins.<{','.join(spins)}>"))
        pro_1 = get_bands_pro_set(xml_data, spin = 0, skipk = skipk, bands_range = bands_range, set_path = set_paths[0])
        pro_2 = get_bands_pro_set(xml_data, spin = 1, skipk = skipk, bands_range = bands_range, set_path = set_paths[1])
        spin_sets = {'u': pro_1.pros[0],'d': pro_2.pros[1]}

    full_dic['spins'] = spin_sets
    full_dic['spins']['labels'] = full_dic['sys_info'].fields
    full_dic['poscar'] = {'SYSTEM':full_dic['sys_info'].SYSTEM,**(get_structure(xml_data).to_dict())}
    return serializer.SpinData(full_dic)

def export_outcar(path=None):
    "Read potential at ionic sites from OUTCAR file."
    if path is None:
        path = './OUTCAR'
    if not os.path.isfile(path):
        raise FileNotFoundError("{} does not exist!".format(path))
    # Raeding it
    with open(r'{}'.format(path),'r') as f:
        lines = f.readlines()
    # Processing
    for i,l in enumerate(lines):
        if 'NIONS' in l:
            N = int(l.split()[-1])
            nlines = np.ceil(N/5).astype(int)
        if 'electrostatic' in l:
            start_index = i+3
            stop_index = start_index+nlines
        if 'fractional' in l:
            first = i+1
        if 'vectors are now' in l:
            b_first = i+5
        if 'NION' in l:
            ion_line = l
        if 'NKPTS' in l:
            kpt_line =l

    NKPTS,NKDIMS,NBANDS = [int(v) for v in re.findall(r"\d+",kpt_line)]
    NEDOS,NIONS = [int(v) for v in re.findall(r"\d+",ion_line)]
    n_kbi = (NKPTS,NBANDS,NIONS)
    # Data manipulation
    # Potential
    data = lines[start_index:stop_index]
    initial = np.loadtxt(StringIO(''.join(data[:-1]))).reshape((-1))
    last = np.loadtxt(StringIO(data[-1]))
    pot_arr = np.hstack([initial,last]).reshape((-1,2))
    pot_arr[:,0] = pot_arr[:,0]-1 # Ion index fixing
    # Nearest neighbors
    pos = lines[first:first+N]
    pos_arr = np.loadtxt(StringIO('\n'.join(pos)))
    pos_arr[pos_arr>0.98] = pos_arr[pos_arr>0.98]-1 # Fixing outer layers
    # positions and potential
    pos_pot = np.hstack([pos_arr,pot_arr[:,1:]])
    basis = np.loadtxt(StringIO(''.join(lines[b_first:b_first+3])))
    final_dict = {'ion_pot':pot_arr,'positions':pos_arr,'site_pot':pos_pot,'basis':basis[:,:3],'rec_basis':basis[:,3:],'n_kbi':n_kbi}
    return serializer.OutcarData(final_dict)

def export_locpot(path:str = None,data_set:str = 0):
    """Returns Data from LOCPOT and similar structure files like CHG/PARCHG etc. Loads only single set based on what is given in data_set argument.
    
    Args:
        locpot (str): path/to/LOCPOT or similar stuructured file like CHG. LOCPOT is auto picked in CWD.
        data_set (int): 0 for electrostatic data, 1 for magnetization data if ISPIN = 2. If non-colinear calculations, 1,2,3 will pick Mx,My,Mz data sets respectively. Only one data set is loaded, so you should know what you are loading.
    
    Returns:
        GridData: ``ipyvasp.serializer.GridData`` object with 3D volumetric data set loaded as attribute 'values'.
        
    Exceptions:
        Would raise index error if magnetization density set is not present in case ``data_set > 0``.
    
    .. note::
        Read `vaspwiki-CHGCAR <https://www.vasp.at/wiki/index.php/CHGCAR>`_ for more info on what data sets are available corresponding to different calculations.
    """
    path = path or './LOCPOT'
    if not os.path.isfile(path):
        raise FileNotFoundError("File {!r} does not exist!".format(path))
    
    if data_set < 0 or data_set > 3:
        raise ValueError("`data_set` should be 0 (for electrostatic),1 (for M or Mx),2 (for My),3 (for Mz)! Got {}".format(data_set))
    
    # data fixing after reading islice from file.
    def fix_data(islice_gen,shape):
        try:
            new_gen = (float(l) for line in islice_gen for l in line.split())
            COUNT = np.prod(shape).astype(int)
            data = np.fromiter(new_gen,dtype=float,count=COUNT) # Count is must for performance
            # data written on LOCPOT is in shape of (NGz,NGy,NGx)
            N_reshape = [shape[2],shape[1],shape[0]]
            data = data.reshape(N_reshape).transpose([2,1,0])
            return data
        except:
            if data_set == 0:
                raise ValueError("File {!r} may not be in proper format!".format(path))
            else:
                raise IndexError("Magnetization density may not be present in {!r}!".format(path))
    
    # Reading File
    with open(path,'r') as f:
        lines = []
        f.seek(0)
        for i in range(8):
            lines.append(f.readline())
        N = sum([int(v) for v in lines[6].split()])
        f.seek(0)
        poscar = []
        for i in range(N+8):
            poscar.append(f.readline())
        f.readline() # Empty one
        Nxyz = [int(v) for v in f.readline().split()] # Grid line read
        nlines = np.ceil(np.prod(Nxyz)/5).astype(int)
        #islice is faster generator for reading potential
        pot_dict = {}
        if data_set == 0:
            pot_dict.update({'values':fix_data(islice(f, nlines),Nxyz)})
            ignore_set = 0 # Pointer already ahead.
        else:
            ignore_set = nlines # Needs to move pointer to magnetization
        #reading Magnetization if True
        ignore_n = np.ceil(N/5).astype(int)+1 #Some kind of useless data
        if data_set == 1:
            print("Note: data_set = 1 picks Mx for non-colinear case, and M for ISPIN = 2.")
            start = ignore_n+ignore_set
            pot_dict.update({'values': fix_data(islice(f, start,start+nlines),Nxyz)})
        elif data_set == 2:
            start = 2*ignore_n+nlines+ignore_set
            pot_dict.update({'values': fix_data(islice(f, start,start+nlines),Nxyz)})
        elif data_set == 3:
            start = 3*ignore_n+2*nlines+ignore_set
            pot_dict.update({'values': fix_data(islice(f, start,start+nlines),Nxyz)})

    # Read Info
    from .sio import export_poscar # Keep inside to avoid import loop
    poscar_data = export_poscar(content = '\n'.join(p.strip() for p in poscar))
    final_dict = dict(SYSTEM = poscar_data.SYSTEM, path = path, **pot_dict, poscar = poscar_data)
    return serializer.GridData(final_dict)
