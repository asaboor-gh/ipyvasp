import re
import os
from io import StringIO
from itertools import islice, chain, product
from collections import namedtuple, Iterable
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
        self._path = os.path.abspath(path) # Keep absolute path in case directory changes
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.path!r})"
    
    @property
    def path(self): return self._path
    
    @property
    def bands(self):
        "Returns a Bands object to access band structure data and plotting methods."
        from .api import Bands
        return Bands(self)
    
    @property
    def dos(self):
        "Returns a Dos object to access density of states data and plotting methods."
        from .api import DOS
        return DOS(self)
    
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
        self._skipk = skipk if isinstance(skipk,(int,np.integer)) else self.get_skipk()
        
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
        info_dict['NKPTS'] = len([v for v in ET.fromstringlist(self.read('<varray.*kpoint','</varray')).iter('v')])
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
     
     
    def get_dos(self, spin = 0, elim = None, atoms = None, orbs = None, gridlim = None):
        # Should be as energy(NE,), total(NE,), integrated(NE,), partial(NE, NATOMS(selected), NORBS) and atoms reference should be returned
        # include vbm property as in evals, just from integrated dos and NELECT
        # check size and if nothing, throw error
        # should have a gridlim like blim to override the elim
        pass 
        
        if atoms and orbs:
            pass
        elif (atoms, orbs) != (None, None):
            raise ValueError('atoms and orbs should be specified together')
    
    def _get_spin_set(self, spin, bands, atoms, orbs, sys_info): 
        "sys_info is the summary data of calculation, used to get the shape of the array."
        if not hasattr(sys_info, 'orbs'):
            raise ValueError(f'The file {self.path!r} does not have projected orbitals!')
            
        shape = (sys_info.NKPTS, sys_info.NBANDS, sys_info.NIONS, len(sys_info.orbs))
        slices = [range(self._skipk, sys_info.NKPTS), bands, atoms, orbs]
            
        gen = (r.strip(' \n/<>r') for r in self.read(f'spin{spin+1}',f'spin{spin+2}|</projected') if '<r>' in r) # stripping is must here to ensure that we get only numbers
        return gen2numpy(gen, shape,slices)

    def get_evals(self, spin = 0, elim = None, atoms = None, orbs = None, blim = None): 
        """
        Returns eigenvalues and occupations of the calculation. If atoms and orbs are specified, then orbitals are included too.
        
        Parameters
        ----------
        spin : int, index of spin set to be pick. Default is 0. Can be 0 or 1.
        elim : tuple, energy range to be returned. Default is None, which returns all eigenvalues. User should supply elim as (min + efermi, max + efermi) to get the correct eigenvalues range.
        atoms : list, indices of atoms to be returned. Default is None, which does not return ionic projections.
        orbs : list, indices of orbitals to be returned. Default is None, which does not return orbitals.
        blim : tuple of length 2, overrides elim and returns eigenvalues in the band range. Useful to load same bands range for spin up and down channels. Plotting classes automatically handle this for spin up and down channels.
        
        Returns
        -------
        Dict2Data object which includes `evals` and `occs` as attributes, and `pros` if atoms and orbs specified.
        """
        info = self.get_summary()
        bands = range(info.NBANDS)
        
        ev = (r.split()[1:3] for r in self.read(f'<set.*spin {spin+1}',f'<set.*spin {spin+2}|</eigenvalues>') if '<r>' in r)
        ev = (e for es in ev for e in es) # flatten
        ev = np.fromiter(ev,float)
        if ev.size: # if not empty
            ev = ev.reshape((-1,info.NBANDS,2))[self._skipk:]
        else:
            raise ValueError('No eigenvalues found for spin {}'.format(spin))
        evals, occs = ev[:,:,0], ev[:,:,1]
        vbm = float(evals[occs > 0.0001].max()) # discard very small occupations
        cbm = float(evals[occs < 0.9999].min()) # discard very small occupations
        evc = (vbm,cbm) # Easy to plot this way like [0, np.ptp(evc)] can work for defult efermi
        
        kvbm = [k for k in np.where(evals == vbm)[0]] # keep as indices here, we don't know the cartesian coordinates of kpoints here
        kcbm = [k for k in np.where(evals == cbm)[0]]
        kvc = tuple(set(product(kvbm,kcbm))) # make unique pairs (by set) of kpoints to plot easily
        
        to_from = (0, -1) # default values
        if blim:
            if (not isinstance(blim, (list, tuple))) and (len(blim) != 2):
                raise TypeError('blim should be a tuple of length 2')
            for b in blim:
                if (not isinstance(b, (int, np.integer))) and (b < 0):
                    raise TypeError('blim should be a tuple of length 2 of positive integers')
            to_from = blim[0], blim[1] + 1 # +1 to include the last index which get skipped in slicing
        elif elim:
            if (not isinstance(elim, (list, tuple))) and (len(elim) != 2):
                raise TypeError('elim should be a tuple of length 2')
            
            idx_max = np.max(np.where(evals[:,:] <= np.max(elim))[1]) + 1
            idx_min = np.min(np.where(evals[:,:] >= np.min(elim))[1])
            to_from = (idx_min, idx_max)
        
        if to_from != (0, -1):
            lo_ind, up_ind = to_from
            evals = evals[:, lo_ind:up_ind]
            occs = occs[:, lo_ind:up_ind]
            bands = range(lo_ind, up_ind)
        
        out = {'spin': spin, 'evals':evals,'occs':occs, 'evc':evc, 'kvc':kvc, 'bands':bands}
        if atoms and orbs:
            out['pros'] = self._get_spin_set(spin, bands, atoms, orbs, info)
            out['atoms'] = atoms
            out['orbs'] = orbs
        elif (atoms, orbs) != (None, None):
            raise ValueError('atoms and orbs should be passed together')
            
        return serializer.Dict2Data(out)

    def get_spins(self, bands = -1, atoms = -1, orbs = -1): 
        # Just have a loop over _get_spin_set and collect all spins, evals must be collected for both spin up and down
        # shape should be (ISPIN, NKPOINTS, NBANDS, [NATOMS(selected), NORBS]) and atoms and bands reference should be returned
        # include vbm property like in evals
        # Tell user they can plot any spin using this data in splots
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
    
    def minify(self):
        "Removes partial dos and projected eigenvalues data from large vasprun.xml to make it smaller in size to save on disk."
        path = os.path.join(os.path.split(self.path)[0], 'mini-vasprun.xml')
        lines_1 = self.read('xml','<partial',skip_last=True)
        lines_2 = islice(self.read('</partial','<projected',skip_last=True), 1, None)
        lines_3 = islice(self.read('</projected','</xml'), 1, None)
        text = ''.join(chain(lines_1, lines_2, lines_3))
        
        with open(path,'w') as f:
            f.write(text)
            
        print('Minified vasprun.xml saved at {}'.format(path))
        

def xml2dict(xmlnode_or_filepath):
    """Convert xml node or xml file content to dictionary. All output text is in string format, so further processing is required to convert into data types/split etc.
    
    Args:
        xmlnode_or_filepath: It is either a path to an xml file or an ``xml.etree.ElementTree.Element`` object.
        
    Each node has ``tag,text,attr,nodes`` attributes. Every text element can be accessed via
    ``xml2dict()['nodes'][index]['nodes'][index]...`` tree which makes it simple.
    """
    if isinstance(xmlnode_or_filepath,str):
        node = ET.parse(xmlnode_or_filepath).getroot()
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


def gen2numpy(gen, shape, slices, raw:bool = False, dtype = float, delimiter = '\s+', include:str = None,exclude:str = '#',fix_format:bool = True):
    """
    Convert a generator of text lines to numpy array while excluding comments, given matches and empty lines. 
    Data is sliced and reshaped as per given shape and slices. It is very efficient for large data files to fetch only required data.
    
    Parameters
    ----------
    gen : generator object. Typical example is `with open('file.txt') as f: gen = itertools.islice(f,0,None)`
    shape : tuple or list of integers. Given shape of data to be read. Last item is considered to be columns. User should keep track of empty lines and excluded lines.
    slices : tuple or list of integers or range or -1. Given slices of data to be read along each dimension. Last item is considered to be columns.
    raw : bool, returns raw data for quick visualizing and determining shape such as columns, if True.
    dtype : data type of numpy array to be returned.
    delimiter : delimiter of data in text file.
    include : string to include in each line to be read. If None, all lines are included.
    exclude : string to exclude in each line to be read. If None, no lines are excluded.
    fix_format : bool, if True, it will fix the format of data in each line. It is useful when data is not properly formatted. like 0.500-0.700 -> 0.500 -0.700
    
    Returns
    -------
    numpy array of given shape and dtype.
    
    Raises
    ------
    Multiple errors are raised if given arguments are not of correct type or shape.
    If number of lines in generators are less than given shape, it will raise ValueError for short iterator from numpy.
    """
    if not isinstance(shape,(list,tuple)):
        raise TypeError(f"shape must be a list/tuple of size of dimensions.")
    
    if not isinstance(slices,(list,tuple)):
        raise TypeError(f"slices must be a list/tuple of size of dimensions.")
    
    if len(shape) != len(slices):
        raise ValueError(f"shape and slices must be of same size.")
    
    for sh in shape:
        if (not isinstance(sh, (int,np.integer))) or (sh < 1):
            raise TypeError(f"Each item in shape must be integer of size of that dimension, at least 1.")
    
    for idx, sli in enumerate(slices):
        if not isinstance(sli, (list, tuple, range, int, np.integer)):
            raise TypeError(f"Expect -1 to pick all data or list/tuple/range to slice data in a dimension, got {sli}")
        if isinstance(sli, (int,np.integer)) and (sli != -1):
            raise TypeError(f"Expect -1 to pick all data or list/tuple/range to slice data in a dimension, got {sli}")
        if isinstance(sli, (list, tuple, range)) and not sli:
            raise TypeError(f"Expect non-empty items in slices, got {type(sli)}")
        
        # Verify order, index and values in take
        if isinstance(sli, (list,tuple,range)):
            if not all(isinstance(i, (int,np.integer)) for i in sli):
                raise TypeError(f"Expect integers in a slice, got {sli}")
            if not all(i >= 0 for i in sli):
                raise ValueError(f"Expect positive integers in a slice, got {sli}")
            if not all(i < shape[idx] for i in sli):
                raise ValueError(f"Some indices in slice {sli} are out of bound for dimension {idx} of size {shape[idx]}")
            
            if not all(a < b for a,b in zip(sli[:-1],sli[1:])): # Check order
                raise ValueError(f"Expect increasing order in a slice, got {sli}")
    
    new_shape = tuple([len(sli) if isinstance(sli, (list, tuple, range)) else N for sli, N in zip(slices, shape)])
    count     = int(np.product(new_shape)) # include columns here for overall data count.
    
    # Process generator    
    if include:
        gen = (l for l in gen if re.search(include,l))

    if exclude:
        gen = (l for l in gen if not re.search(exclude,l))

    gen = (l for l in gen if l.strip()) # remove empty lines
    
    gen = islice(gen, 0, int(np.product(shape[:-1]))) # Discard lines after required data
    
    def fold_dim(gen, take, N):
        if take == -1:
            yield from gen # return does not work here.
        
        j = 0
        group = ()
        for i, line in enumerate(gen, start =1):
            if j in take:
                group = chain(group, (line,))
            j = j + 1
            if i % N == 0:
                j = 0
                yield group
                group = ()
    
    def flatten(gen):
        for g in gen:
            if isinstance(g, Iterable) and not isinstance(g, str):
                yield from flatten(g)
            else:
                yield g
    
    # Slice data, but we keep columns here for now, should return raw data if asked.  
    for take, N in zip(slices[:-1][::-1], shape[:-1][::-1]): # reverse order to pick innermost first but leave columns
        gen = fold_dim(gen, take, N)
    else: # flatter on success
        gen = flatten(gen)
    
    # Negative connected digits to avoid fix after slicing to reduce time.
    if fix_format:
        gen = (re.sub(r"(\d)-(\d)",r"\1 -\2",l) for l in gen)
        
    if raw:
        return ''.join(gen) # lines already have '\n' at the end.
    
    # Split columns and flatten after escaped from raw return.
    gen = (item for line in gen for item in line.replace(delimiter,'  ').split())
    gen = flatten(fold_dim(gen, slices[-1], shape[-1])) # columns

    data = np.fromiter(gen,dtype = dtype, count = count)
    return data.reshape(new_shape)


def minify_vasprun(path : str):
    "Minify vasprun.xml file by removing projected data."
    return Vasprun(path).minify()


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
