??5
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??3
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*@*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	?*@*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*O
shared_name@>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel
?
Rbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/Read/ReadVariableOpReadVariableOp>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel*
_output_shapes

:@@*
dtype0
?
Hbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*Y
shared_nameJHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel
?
\bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOpHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel*
_output_shapes

:@@*
dtype0
?
<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*M
shared_name><bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias
?
Pbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/Read/ReadVariableOpReadVariableOp<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias*
_output_shapes
:@*
dtype0
?
?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*P
shared_nameA?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel
?
Sbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/Read/ReadVariableOpReadVariableOp?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel*
_output_shapes

:@@*
dtype0
?
Ibidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*Z
shared_nameKIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel
?
]bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/Read/ReadVariableOpReadVariableOpIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel*
_output_shapes

:@@*
dtype0
?
=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*N
shared_name?=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias
?
Qbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/Read/ReadVariableOpReadVariableOp=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*@*.
shared_nameAdam/embedding_1/embeddings/m
?
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes
:	?*@*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	?@*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
EAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*V
shared_nameGEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/m
?
YAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/m/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/m*
_output_shapes

:@@*
dtype0
?
OAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*`
shared_nameQOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/m
?
cAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
?
CAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*T
shared_nameECAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/m
?
WAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/m/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/m*
_output_shapes
:@*
dtype0
?
FAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*W
shared_nameHFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/m
?
ZAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/m*
_output_shapes

:@@*
dtype0
?
PAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*a
shared_nameRPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/m
?
dAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
?
DAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*U
shared_nameFDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/m
?
XAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/m/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/m*
_output_shapes
:@*
dtype0
?
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*@*.
shared_nameAdam/embedding_1/embeddings/v
?
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes
:	?*@*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	?@*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
EAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*V
shared_nameGEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/v
?
YAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/v/Read/ReadVariableOpReadVariableOpEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/v*
_output_shapes

:@@*
dtype0
?
OAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*`
shared_nameQOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/v
?
cAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
?
CAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*T
shared_nameECAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/v
?
WAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/v/Read/ReadVariableOpReadVariableOpCAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/v*
_output_shapes
:@*
dtype0
?
FAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*W
shared_nameHFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/v
?
ZAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/v*
_output_shapes

:@@*
dtype0
?
PAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*a
shared_nameRPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/v
?
dAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
?
DAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*U
shared_nameFDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/v
?
XAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/v/Read/ReadVariableOpReadVariableOpDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
?H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?G B?G
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
y
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratem{m|m}m~m'm?(m?)m?*m?+m?,m?v?v?v?v?v?'v?(v?)v?*v?+v?,v?
N
0
'1
(2
)3
*4
+5
,6
7
8
9
10
N
0
'1
(2
)3
*4
+5
,6
7
8
9
10
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
l
7cell
8
state_spec
9	variables
:trainable_variables
;regularization_losses
<	keras_api
l
=cell
>
state_spec
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
*
'0
(1
)2
*3
+4
,5
*
'0
(1
)2
*3
+4
,5
 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
 regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

R0
S1
 
 
 
 
 
 
 
~

'kernel
(recurrent_kernel
)bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
 

'0
(1
)2

'0
(1
)2
 
?

Xstates
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
9	variables
:trainable_variables
;regularization_losses
~

*kernel
+recurrent_kernel
,bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
 

*0
+1
,2

*0
+1
,2
 
?

bstates
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
?	variables
@trainable_variables
Aregularization_losses
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	htotal
	icount
j	variables
k	keras_api
D
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api

'0
(1
)2

'0
(1
)2
 
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
 
 

70
 
 
 

*0
+1
,2

*0
+1
,2
 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
^	variables
_trainable_variables
`regularization_losses
 
 

=0
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

j	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

o	variables
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUECAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUECAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_embedding_1_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddings>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/biasHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/biasIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kerneldense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_392029
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpRbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/Read/ReadVariableOp\bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/Read/ReadVariableOpPbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/Read/ReadVariableOpSbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/Read/ReadVariableOp]bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/Read/ReadVariableOpQbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOpYAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/m/Read/ReadVariableOpcAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/m/Read/ReadVariableOpWAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/m/Read/ReadVariableOpZAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/m/Read/ReadVariableOpdAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/m/Read/ReadVariableOpXAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpYAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/v/Read/ReadVariableOpcAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/v/Read/ReadVariableOpWAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/v/Read/ReadVariableOpZAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/v/Read/ReadVariableOpdAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/v/Read/ReadVariableOpXAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_394982
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernelHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernelIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/biastotalcounttotal_1count_1Adam/embedding_1/embeddings/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/mOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/mCAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/mFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/mPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/mDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/mAdam/embedding_1/embeddings/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vEAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/vOAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/vCAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/vFAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/vPAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/vDAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_395118ו2
?,
?
while_body_394159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_10/MatMul/ReadVariableOp?0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?i
?
6bidirectional_1_forward_simple_rnn_1_while_body_392429f
bbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_loop_counterl
hbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations:
6bidirectional_1_forward_simple_rnn_1_while_placeholder<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_1<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_2<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_3e
abidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1_0?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0p
^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@m
_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@r
`bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@7
3bidirectional_1_forward_simple_rnn_1_while_identity9
5bidirectional_1_forward_simple_rnn_1_while_identity_19
5bidirectional_1_forward_simple_rnn_1_while_identity_29
5bidirectional_1_forward_simple_rnn_1_while_identity_39
5bidirectional_1_forward_simple_rnn_1_while_identity_49
5bidirectional_1_forward_simple_rnn_1_while_identity_5c
_bidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensorn
\bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@k
]bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@p
^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Tbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Sbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Ubidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
\bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Nbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_06bidirectional_1_forward_simple_rnn_1_while_placeholderebidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
^bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Pbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_06bidirectional_1_forward_simple_rnn_1_while_placeholdergbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Sbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Dbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulUbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0[bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Tbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
Ebidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAddNbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0\bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ubidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp`bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Fbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul8bidirectional_1_forward_simple_rnn_1_while_placeholder_3]bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Abidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2Nbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0Pbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Bbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanhEbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
9bidirectional_1/forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/bidirectional_1/forward_simple_rnn_1/while/TileTileWbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Bbidirectional_1/forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
3bidirectional_1/forward_simple_rnn_1/while/SelectV2SelectV28bidirectional_1/forward_simple_rnn_1/while/Tile:output:0Fbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:08bidirectional_1_forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@?
;bidirectional_1/forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
1bidirectional_1/forward_simple_rnn_1/while/Tile_1TileWbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Dbidirectional_1/forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
5bidirectional_1/forward_simple_rnn_1/while/SelectV2_1SelectV2:bidirectional_1/forward_simple_rnn_1/while/Tile_1:output:0Fbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:08bidirectional_1_forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
Obidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem8bidirectional_1_forward_simple_rnn_1_while_placeholder_16bidirectional_1_forward_simple_rnn_1_while_placeholder<bidirectional_1/forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???r
0bidirectional_1/forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
.bidirectional_1/forward_simple_rnn_1/while/addAddV26bidirectional_1_forward_simple_rnn_1_while_placeholder9bidirectional_1/forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: t
2bidirectional_1/forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
0bidirectional_1/forward_simple_rnn_1/while/add_1AddV2bbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_loop_counter;bidirectional_1/forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
3bidirectional_1/forward_simple_rnn_1/while/IdentityIdentity4bidirectional_1/forward_simple_rnn_1/while/add_1:z:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_1Identityhbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations0^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_2Identity2bidirectional_1/forward_simple_rnn_1/while/add:z:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_3Identity_bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_4Identity<bidirectional_1/forward_simple_rnn_1/while/SelectV2:output:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
5bidirectional_1/forward_simple_rnn_1/while/Identity_5Identity>bidirectional_1/forward_simple_rnn_1/while/SelectV2_1:output:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
/bidirectional_1/forward_simple_rnn_1/while/NoOpNoOpU^bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpT^bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpV^bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
_bidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1abidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1_0"s
3bidirectional_1_forward_simple_rnn_1_while_identity<bidirectional_1/forward_simple_rnn_1/while/Identity:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_1>bidirectional_1/forward_simple_rnn_1/while/Identity_1:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_2>bidirectional_1/forward_simple_rnn_1/while/Identity_2:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_3>bidirectional_1/forward_simple_rnn_1/while/Identity_3:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_4>bidirectional_1/forward_simple_rnn_1/while/Identity_4:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_5>bidirectional_1/forward_simple_rnn_1/while/Identity_5:output:0"?
]bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource`bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
\bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Tbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpTbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Sbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpSbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Ubidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpUbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&forward_simple_rnn_1_while_cond_392785F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2H
Dforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_392785___redundant_placeholder0^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_392785___redundant_placeholder1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_392785___redundant_placeholder2^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_392785___redundant_placeholder3'
#forward_simple_rnn_1_while_identity
?
forward_simple_rnn_1/while/LessLess&forward_simple_rnn_1_while_placeholderDforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_1/while/IdentityIdentity#forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?

?
0__inference_bidirectional_1_layer_call_fn_392744

inputs
mask

unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391794p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????????????@:??????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
5__inference_forward_simple_rnn_1_layer_call_fn_393793

inputs
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_391078o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
while_cond_393942
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_393942___redundant_placeholder04
0while_while_cond_393942___redundant_placeholder14
0while_while_cond_393942___redundant_placeholder24
0while_while_cond_393942___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390322

inputs

states0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?,
?
while_body_391012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_10/MatMul/ReadVariableOp?0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?	
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_392674

inputs*
embedding_lookup_392668:	?*@
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_392668Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/392668*4
_output_shapes"
 :??????????????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/392668*4
_output_shapes"
 :??????????????????@?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
'backward_simple_rnn_1_while_cond_393359H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3J
Fbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393359___redundant_placeholder0`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393359___redundant_placeholder1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393359___redundant_placeholder2`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393359___redundant_placeholder3`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393359___redundant_placeholder4(
$backward_simple_rnn_1_while_identity
?
 backward_simple_rnn_1/while/LessLess'backward_simple_rnn_1_while_placeholderFbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_1/while/IdentityIdentity$backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
&forward_simple_rnn_1_while_cond_393001F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2H
Dforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393001___redundant_placeholder0^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393001___redundant_placeholder1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393001___redundant_placeholder2^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393001___redundant_placeholder3'
#forward_simple_rnn_1_while_identity
?
forward_simple_rnn_1/while/LessLess&forward_simple_rnn_1_while_placeholderDforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_1/while/IdentityIdentity#forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?>
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394225

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_10/BiasAdd/ReadVariableOp?(simple_rnn_cell_10/MatMul/ReadVariableOp?*simple_rnn_cell_10/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_394159*
condR
while_cond_394158*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390442

inputs

states0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?

?
0__inference_bidirectional_1_layer_call_fn_392726

inputs
mask

unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391413p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????????????@:??????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
'backward_simple_rnn_1_while_cond_391329H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3J
Fbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391329___redundant_placeholder0`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391329___redundant_placeholder1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391329___redundant_placeholder2`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391329___redundant_placeholder3`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391329___redundant_placeholder4(
$backward_simple_rnn_1_while_identity
?
 backward_simple_rnn_1/while/LessLess'backward_simple_rnn_1_while_placeholderFbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_1/while/IdentityIdentity$backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?>
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394009
inputs_0C
1simple_rnn_cell_10_matmul_readvariableop_resource:@@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_10/BiasAdd/ReadVariableOp?(simple_rnn_cell_10/MatMul/ReadVariableOp?*simple_rnn_cell_10/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_393943*
condR
while_cond_393942*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?,
?
while_body_393835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_10/MatMul/ReadVariableOp?0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_390334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390334___redundant_placeholder04
0while_while_cond_390334___redundant_placeholder14
0while_while_cond_390334___redundant_placeholder24
0while_while_cond_390334___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?

7bidirectional_1_backward_simple_rnn_1_while_cond_392560h
dbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_loop_countern
jbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations;
7bidirectional_1_backward_simple_rnn_1_while_placeholder=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_1=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_2=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_3j
fbidirectional_1_backward_simple_rnn_1_while_less_bidirectional_1_backward_simple_rnn_1_strided_slice_1?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392560___redundant_placeholder0?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392560___redundant_placeholder1?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392560___redundant_placeholder2?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392560___redundant_placeholder3?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392560___redundant_placeholder48
4bidirectional_1_backward_simple_rnn_1_while_identity
?
0bidirectional_1/backward_simple_rnn_1/while/LessLess7bidirectional_1_backward_simple_rnn_1_while_placeholderfbidirectional_1_backward_simple_rnn_1_while_less_bidirectional_1_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: ?
4bidirectional_1/backward_simple_rnn_1/while/IdentityIdentity4bidirectional_1/backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "u
4bidirectional_1_backward_simple_rnn_1_while_identity=bidirectional_1/backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?j
?
7bidirectional_1_backward_simple_rnn_1_while_body_392561h
dbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_loop_countern
jbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations;
7bidirectional_1_backward_simple_rnn_1_while_placeholder=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_1=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_2=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_3g
cbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1_0?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0q
_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@n
`bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@s
abidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@8
4bidirectional_1_backward_simple_rnn_1_while_identity:
6bidirectional_1_backward_simple_rnn_1_while_identity_1:
6bidirectional_1_backward_simple_rnn_1_while_identity_2:
6bidirectional_1_backward_simple_rnn_1_while_identity_3:
6bidirectional_1_backward_simple_rnn_1_while_identity_4:
6bidirectional_1_backward_simple_rnn_1_while_identity_5e
abidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensoro
]bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@l
^bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@q
_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ubidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Tbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Vbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
]bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Obidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_07bidirectional_1_backward_simple_rnn_1_while_placeholderfbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
_bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Qbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_07bidirectional_1_backward_simple_rnn_1_while_placeholderhbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Tbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Ebidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulVbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0\bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ubidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp`bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
Fbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAddObidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0]bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Vbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpabidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Gbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul9bidirectional_1_backward_simple_rnn_1_while_placeholder_3^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Bbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2Obidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Qbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Cbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanhFbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
:bidirectional_1/backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
0bidirectional_1/backward_simple_rnn_1/while/TileTileXbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Cbidirectional_1/backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
4bidirectional_1/backward_simple_rnn_1/while/SelectV2SelectV29bidirectional_1/backward_simple_rnn_1/while/Tile:output:0Gbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:09bidirectional_1_backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@?
<bidirectional_1/backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
2bidirectional_1/backward_simple_rnn_1/while/Tile_1TileXbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Ebidirectional_1/backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
6bidirectional_1/backward_simple_rnn_1/while/SelectV2_1SelectV2;bidirectional_1/backward_simple_rnn_1/while/Tile_1:output:0Gbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:09bidirectional_1_backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
Pbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9bidirectional_1_backward_simple_rnn_1_while_placeholder_17bidirectional_1_backward_simple_rnn_1_while_placeholder=bidirectional_1/backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???s
1bidirectional_1/backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
/bidirectional_1/backward_simple_rnn_1/while/addAddV27bidirectional_1_backward_simple_rnn_1_while_placeholder:bidirectional_1/backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: u
3bidirectional_1/backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
1bidirectional_1/backward_simple_rnn_1/while/add_1AddV2dbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_loop_counter<bidirectional_1/backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
4bidirectional_1/backward_simple_rnn_1/while/IdentityIdentity5bidirectional_1/backward_simple_rnn_1/while/add_1:z:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_1Identityjbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations1^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_2Identity3bidirectional_1/backward_simple_rnn_1/while/add:z:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_3Identity`bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_4Identity=bidirectional_1/backward_simple_rnn_1/while/SelectV2:output:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
6bidirectional_1/backward_simple_rnn_1/while/Identity_5Identity?bidirectional_1/backward_simple_rnn_1/while/SelectV2_1:output:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
0bidirectional_1/backward_simple_rnn_1/while/NoOpNoOpV^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpU^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpW^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
abidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1cbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1_0"u
4bidirectional_1_backward_simple_rnn_1_while_identity=bidirectional_1/backward_simple_rnn_1/while/Identity:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_1?bidirectional_1/backward_simple_rnn_1/while/Identity_1:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_2?bidirectional_1/backward_simple_rnn_1/while/Identity_2:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_3?bidirectional_1/backward_simple_rnn_1/while/Identity_3:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_4?bidirectional_1/backward_simple_rnn_1/while/Identity_4:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_5?bidirectional_1/backward_simple_rnn_1/while/Identity_5:output:0"?
^bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource`bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceabidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
]bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Ubidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpUbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Tbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpTbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Vbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpVbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_390495
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390495___redundant_placeholder04
0while_while_cond_390495___redundant_placeholder14
0while_while_cond_390495___redundant_placeholder24
0while_while_cond_390495___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_391878

inputs%
embedding_1_391849:	?*@(
bidirectional_1_391854:@@$
bidirectional_1_391856:@(
bidirectional_1_391858:@@(
bidirectional_1_391860:@@$
bidirectional_1_391862:@(
bidirectional_1_391864:@@!
dense_2_391867:	?@
dense_2_391869:@ 
dense_3_391872:@
dense_3_391874:
identity??'bidirectional_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_391849*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_391141[
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0bidirectional_1_391854bidirectional_1_391856bidirectional_1_391858bidirectional_1_391860bidirectional_1_391862bidirectional_1_391864*
Tin

2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391794?
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_391867dense_2_391869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_391438?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_391872dense_3_391874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_391454w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^bidirectional_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883?
~sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_loop_counter?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_maximum_iterationsH
Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholderJ
Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_1J
Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_2J
Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_3?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_less_sequential_1_bidirectional_1_backward_simple_rnn_1_strided_slice_1?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883___redundant_placeholder0?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883___redundant_placeholder1?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883___redundant_placeholder2?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883___redundant_placeholder3?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883___redundant_placeholder4E
Asequential_1_bidirectional_1_backward_simple_rnn_1_while_identity
?
=sequential_1/bidirectional_1/backward_simple_rnn_1/while/LessLessDsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder?sequential_1_bidirectional_1_backward_simple_rnn_1_while_less_sequential_1_bidirectional_1_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: ?
Asequential_1/bidirectional_1/backward_simple_rnn_1/while/IdentityIdentityAsequential_1/bidirectional_1/backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "?
Asequential_1_bidirectional_1_backward_simple_rnn_1_while_identityJsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
while_cond_394532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_394532___redundant_placeholder04
0while_while_cond_394532___redundant_placeholder14
0while_while_cond_394532___redundant_placeholder24
0while_while_cond_394532___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?4
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390263

inputs+
simple_rnn_cell_10_390188:@@'
simple_rnn_cell_10_390190:@+
simple_rnn_cell_10_390192:@@
identity??*simple_rnn_cell_10/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
*simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_10_390188simple_rnn_cell_10_390190simple_rnn_cell_10_390192*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390148n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_10_390188simple_rnn_cell_10_390190simple_rnn_cell_10_390192*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390200*
condR
while_cond_390199*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@{
NoOpNoOp+^simple_rnn_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2X
*simple_rnn_cell_10/StatefulPartitionedCall*simple_rnn_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?>
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_393901
inputs_0C
1simple_rnn_cell_10_matmul_readvariableop_resource:@@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_10/BiasAdd/ReadVariableOp?(simple_rnn_cell_10/MatMul/ReadVariableOp?*simple_rnn_cell_10/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_393835*
condR
while_cond_393834*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
while_cond_394642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_394642___redundant_placeholder04
0while_while_cond_394642___redundant_placeholder14
0while_while_cond_394642___redundant_placeholder24
0while_while_cond_394642___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?U
?
&forward_simple_rnn_1_while_body_391198F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3E
Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0?
}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@]
Oforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_1_while_identity)
%forward_simple_rnn_1_while_identity_1)
%forward_simple_rnn_1_while_identity_2)
%forward_simple_rnn_1_while_identity_3)
%forward_simple_rnn_1_while_identity_4)
%forward_simple_rnn_1_while_identity_5C
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor^
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@[
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
Lforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
>forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderUforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Nforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderWforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
4forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulEforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
5forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd>forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0Lforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
6forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul(forward_simple_rnn_1_while_placeholder_3Mforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2>forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0@forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanh5forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@z
)forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
forward_simple_rnn_1/while/TileTileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:02forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
#forward_simple_rnn_1/while/SelectV2SelectV2(forward_simple_rnn_1/while/Tile:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@|
+forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
!forward_simple_rnn_1/while/Tile_1TileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:04forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
%forward_simple_rnn_1/while/SelectV2_1SelectV2*forward_simple_rnn_1/while/Tile_1:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_1_while_placeholder_1&forward_simple_rnn_1_while_placeholder,forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???b
 forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
forward_simple_rnn_1/while/addAddV2&forward_simple_rnn_1_while_placeholder)forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
 forward_simple_rnn_1/while/add_1AddV2Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counter+forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
#forward_simple_rnn_1/while/IdentityIdentity$forward_simple_rnn_1/while/add_1:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_1IdentityHforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_2Identity"forward_simple_rnn_1/while/add:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_3IdentityOforward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_4Identity,forward_simple_rnn_1/while/SelectV2:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
%forward_simple_rnn_1/while/Identity_5Identity.forward_simple_rnn_1/while/SelectV2_1:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
forward_simple_rnn_1/while/NoOpNoOpE^forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpF^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0"S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0"W
%forward_simple_rnn_1_while_identity_1.forward_simple_rnn_1/while/Identity_1:output:0"W
%forward_simple_rnn_1_while_identity_2.forward_simple_rnn_1/while/Identity_2:output:0"W
%forward_simple_rnn_1_while_identity_3.forward_simple_rnn_1/while/Identity_3:output:0"W
%forward_simple_rnn_1_while_identity_4.forward_simple_rnn_1/while/Identity_4:output:0"W
%forward_simple_rnn_1_while_identity_5.forward_simple_rnn_1/while/Identity_5:output:0"?
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourceOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcePforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpDforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpCforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpEforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_391438

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_390732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390732___redundant_placeholder04
0while_while_cond_390732___redundant_placeholder14
0while_while_cond_390732___redundant_placeholder24
0while_while_cond_390732___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
'backward_simple_rnn_1_while_cond_391710H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3J
Fbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391710___redundant_placeholder0`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391710___redundant_placeholder1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391710___redundant_placeholder2`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391710___redundant_placeholder3`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_391710___redundant_placeholder4(
$backward_simple_rnn_1_while_identity
?
 backward_simple_rnn_1/while/LessLess'backward_simple_rnn_1_while_placeholderFbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_1/while/IdentityIdentity$backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?@
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390799

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource:@@@
2simple_rnn_cell_11_biasadd_readvariableop_resource:@E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_11/BiasAdd/ReadVariableOp?(simple_rnn_cell_11/MatMul/ReadVariableOp?*simple_rnn_cell_11/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'????????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_11/TanhTanhsimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390733*
condR
while_cond_390732*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ԧ
?
!__inference__wrapped_model_389980
embedding_1_inputC
0sequential_1_embedding_1_embedding_lookup_389697:	?*@u
csequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@r
dsequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@w
esequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@v
dsequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@s
esequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@x
fsequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@F
3sequential_1_dense_2_matmul_readvariableop_resource:	?@B
4sequential_1_dense_2_biasadd_readvariableop_resource:@E
3sequential_1_dense_3_matmul_readvariableop_resource:@B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity??\sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?[sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?]sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?8sequential_1/bidirectional_1/backward_simple_rnn_1/while?[sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Zsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp?\sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?7sequential_1/bidirectional_1/forward_simple_rnn_1/while?+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?*sequential_1/dense_3/MatMul/ReadVariableOp?)sequential_1/embedding_1/embedding_lookup?
sequential_1/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
)sequential_1/embedding_1/embedding_lookupResourceGather0sequential_1_embedding_1_embedding_lookup_389697!sequential_1/embedding_1/Cast:y:0*
Tindices0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/389697*4
_output_shapes"
 :??????????????????@*
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/389697*4
_output_shapes"
 :??????????????????@?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@h
#sequential_1/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!sequential_1/embedding_1/NotEqualNotEqualembedding_1_input,sequential_1/embedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
7sequential_1/bidirectional_1/forward_simple_rnn_1/ShapeShape=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Esequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_1/bidirectional_1/forward_simple_rnn_1/strided_sliceStridedSlice@sequential_1/bidirectional_1/forward_simple_rnn_1/Shape:output:0Nsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice/stack:output:0Psequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice/stack_1:output:0Psequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_1/bidirectional_1/forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
>sequential_1/bidirectional_1/forward_simple_rnn_1/zeros/packedPackHsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice:output:0Isequential_1/bidirectional_1/forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
=sequential_1/bidirectional_1/forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7sequential_1/bidirectional_1/forward_simple_rnn_1/zerosFillGsequential_1/bidirectional_1/forward_simple_rnn_1/zeros/packed:output:0Fsequential_1/bidirectional_1/forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@?
@sequential_1/bidirectional_1/forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
;sequential_1/bidirectional_1/forward_simple_rnn_1/transpose	Transpose=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0Isequential_1/bidirectional_1/forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
9sequential_1/bidirectional_1/forward_simple_rnn_1/Shape_1Shape?sequential_1/bidirectional_1/forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:?
Gsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Isequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Isequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1StridedSliceBsequential_1/bidirectional_1/forward_simple_rnn_1/Shape_1:output:0Psequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack:output:0Rsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_1:output:0Rsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_1/bidirectional_1/forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<sequential_1/bidirectional_1/forward_simple_rnn_1/ExpandDims
ExpandDims%sequential_1/embedding_1/NotEqual:z:0Isequential_1/bidirectional_1/forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :???????????????????
Bsequential_1/bidirectional_1/forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
=sequential_1/bidirectional_1/forward_simple_rnn_1/transpose_1	TransposeEsequential_1/bidirectional_1/forward_simple_rnn_1/ExpandDims:output:0Ksequential_1/bidirectional_1/forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :???????????????????
Msequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?sequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2TensorListReserveVsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2/element_shape:output:0Jsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
gsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Ysequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_1/bidirectional_1/forward_simple_rnn_1/transpose:y:0psequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Gsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Isequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Isequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2StridedSlice?sequential_1/bidirectional_1/forward_simple_rnn_1/transpose:y:0Psequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack:output:0Rsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_1:output:0Rsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Zsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpcsequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Ksequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMulJsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_2:output:0bsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
[sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpdsequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Lsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAddUsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0csequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
\sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpesequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Msequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul@sequential_1/bidirectional_1/forward_simple_rnn_1/zeros:output:0dsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Hsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/addAddV2Usequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0Wsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Isequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/TanhTanhLsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
Osequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Asequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1TensorListReserveXsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0Jsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???x
6sequential_1/bidirectional_1/forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : ?
Osequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Asequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_2TensorListReserveXsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0Jsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
isequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
[sequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorAsequential_1/bidirectional_1/forward_simple_rnn_1/transpose_1:y:0rsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
<sequential_1/bidirectional_1/forward_simple_rnn_1/zeros_like	ZerosLikeMsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@?
Jsequential_1/bidirectional_1/forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Dsequential_1/bidirectional_1/forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
7sequential_1/bidirectional_1/forward_simple_rnn_1/whileWhileMsequential_1/bidirectional_1/forward_simple_rnn_1/while/loop_counter:output:0Ssequential_1/bidirectional_1/forward_simple_rnn_1/while/maximum_iterations:output:0?sequential_1/bidirectional_1/forward_simple_rnn_1/time:output:0Jsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1:handle:0@sequential_1/bidirectional_1/forward_simple_rnn_1/zeros_like:y:0@sequential_1/bidirectional_1/forward_simple_rnn_1/zeros:output:0Jsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0isequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0ksequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0csequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourcedsequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceesequential_1_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *O
bodyGRE
Csequential_1_bidirectional_1_forward_simple_rnn_1_while_body_389752*O
condGRE
Csequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
bsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Tsequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack@sequential_1/bidirectional_1/forward_simple_rnn_1/while:output:3ksequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0?
Gsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Isequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Isequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3StridedSlice]sequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Psequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack:output:0Rsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_1:output:0Rsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Bsequential_1/bidirectional_1/forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
=sequential_1/bidirectional_1/forward_simple_rnn_1/transpose_2	Transpose]sequential_1/bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_1/bidirectional_1/forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
8sequential_1/bidirectional_1/backward_simple_rnn_1/ShapeShape=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Fsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Hsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_1/bidirectional_1/backward_simple_rnn_1/strided_sliceStridedSliceAsequential_1/bidirectional_1/backward_simple_rnn_1/Shape:output:0Osequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice/stack:output:0Qsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice/stack_1:output:0Qsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_1/bidirectional_1/backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
?sequential_1/bidirectional_1/backward_simple_rnn_1/zeros/packedPackIsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice:output:0Jsequential_1/bidirectional_1/backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
>sequential_1/bidirectional_1/backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8sequential_1/bidirectional_1/backward_simple_rnn_1/zerosFillHsequential_1/bidirectional_1/backward_simple_rnn_1/zeros/packed:output:0Gsequential_1/bidirectional_1/backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@?
Asequential_1/bidirectional_1/backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
<sequential_1/bidirectional_1/backward_simple_rnn_1/transpose	Transpose=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0Jsequential_1/bidirectional_1/backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
:sequential_1/bidirectional_1/backward_simple_rnn_1/Shape_1Shape@sequential_1/bidirectional_1/backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:?
Hsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1StridedSliceCsequential_1/bidirectional_1/backward_simple_rnn_1/Shape_1:output:0Qsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack:output:0Ssequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_1:output:0Ssequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_1/bidirectional_1/backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
=sequential_1/bidirectional_1/backward_simple_rnn_1/ExpandDims
ExpandDims%sequential_1/embedding_1/NotEqual:z:0Jsequential_1/bidirectional_1/backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :???????????????????
Csequential_1/bidirectional_1/backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
>sequential_1/bidirectional_1/backward_simple_rnn_1/transpose_1	TransposeFsequential_1/bidirectional_1/backward_simple_rnn_1/ExpandDims:output:0Lsequential_1/bidirectional_1/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :???????????????????
Nsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
@sequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2TensorListReserveWsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2/element_shape:output:0Ksequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Asequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2	ReverseV2@sequential_1/bidirectional_1/backward_simple_rnn_1/transpose:y:0Jsequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
hsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Zsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEsequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2:output:0qsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Hsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2StridedSlice@sequential_1/bidirectional_1/backward_simple_rnn_1/transpose:y:0Qsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack:output:0Ssequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_1:output:0Ssequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
[sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpdsequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Lsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMulKsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_2:output:0csequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
\sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpesequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Msequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAddVsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0dsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
]sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpfsequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Nsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMulAsequential_1/bidirectional_1/backward_simple_rnn_1/zeros:output:0esequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Isequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/addAddV2Vsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0Xsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Jsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/TanhTanhMsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
Psequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Bsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1TensorListReserveYsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0Ksequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
7sequential_1/bidirectional_1/backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
>sequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2_1	ReverseV2Bsequential_1/bidirectional_1/backward_simple_rnn_1/transpose_1:y:0Lsequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :???????????????????
Psequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Bsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_2TensorListReserveYsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0Ksequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
jsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
\sequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorGsequential_1/bidirectional_1/backward_simple_rnn_1/ReverseV2_1:output:0ssequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
=sequential_1/bidirectional_1/backward_simple_rnn_1/zeros_like	ZerosLikeNsequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@?
Ksequential_1/bidirectional_1/backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Esequential_1/bidirectional_1/backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
8sequential_1/bidirectional_1/backward_simple_rnn_1/whileWhileNsequential_1/bidirectional_1/backward_simple_rnn_1/while/loop_counter:output:0Tsequential_1/bidirectional_1/backward_simple_rnn_1/while/maximum_iterations:output:0@sequential_1/bidirectional_1/backward_simple_rnn_1/time:output:0Ksequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1:handle:0Asequential_1/bidirectional_1/backward_simple_rnn_1/zeros_like:y:0Asequential_1/bidirectional_1/backward_simple_rnn_1/zeros:output:0Ksequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0jsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0lsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0dsequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceesequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourcefsequential_1_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *P
bodyHRF
Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_body_389884*P
condHRF
Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_cond_389883*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
csequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Usequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackAsequential_1/bidirectional_1/backward_simple_rnn_1/while:output:3lsequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0?
Hsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3StridedSlice^sequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Qsequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack:output:0Ssequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_1:output:0Ssequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Csequential_1/bidirectional_1/backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
>sequential_1/bidirectional_1/backward_simple_rnn_1/transpose_2	Transpose^sequential_1/bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Lsequential_1/bidirectional_1/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@j
(sequential_1/bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_1/bidirectional_1/concatConcatV2Jsequential_1/bidirectional_1/forward_simple_rnn_1/strided_slice_3:output:0Ksequential_1/bidirectional_1/backward_simple_rnn_1/strided_slice_3:output:01sequential_1/bidirectional_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_1/dense_2/MatMulMatMul,sequential_1/bidirectional_1/concat:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@z
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp]^sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp\^sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp^^sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp9^sequential_1/bidirectional_1/backward_simple_rnn_1/while\^sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp[^sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp]^sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp8^sequential_1/bidirectional_1/forward_simple_rnn_1/while,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2?
\sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp\sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
[sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp[sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
]sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp]sequential_1/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2t
8sequential_1/bidirectional_1/backward_simple_rnn_1/while8sequential_1/bidirectional_1/backward_simple_rnn_1/while2?
[sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp[sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Zsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpZsequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
\sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp\sequential_1/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp2r
7sequential_1/bidirectional_1/forward_simple_rnn_1/while7sequential_1/bidirectional_1/forward_simple_rnn_1/while2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
?

?
-__inference_sequential_1_layer_call_fn_391930
embedding_1_input
unknown:	?*@
	unknown_0:@@
	unknown_1:@
	unknown_2:@@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:	?@
	unknown_7:@
	unknown_8:@
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_391878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_391454

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
&forward_simple_rnn_1_while_cond_393227F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3H
Dforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393227___redundant_placeholder0^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393227___redundant_placeholder1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393227___redundant_placeholder2^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393227___redundant_placeholder3^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393227___redundant_placeholder4'
#forward_simple_rnn_1_while_identity
?
forward_simple_rnn_1/while/LessLess&forward_simple_rnn_1_while_placeholderDforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_1/while/IdentityIdentity#forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
6__inference_backward_simple_rnn_1_layer_call_fn_394269

inputs
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_390810

inputs-
forward_simple_rnn_1_390683:@@)
forward_simple_rnn_1_390685:@-
forward_simple_rnn_1_390687:@@.
backward_simple_rnn_1_390800:@@*
backward_simple_rnn_1_390802:@.
backward_simple_rnn_1_390804:@@
identity??-backward_simple_rnn_1/StatefulPartitionedCall?,forward_simple_rnn_1/StatefulPartitionedCall?
,forward_simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_1_390683forward_simple_rnn_1_390685forward_simple_rnn_1_390687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390682?
-backward_simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_1_390800backward_simple_rnn_1_390802backward_simple_rnn_1_390804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390799M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV25forward_simple_rnn_1/StatefulPartitionedCall:output:06backward_simple_rnn_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp.^backward_simple_rnn_1/StatefulPartitionedCall-^forward_simple_rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2^
-backward_simple_rnn_1/StatefulPartitionedCall-backward_simple_rnn_1/StatefulPartitionedCall2\
,forward_simple_rnn_1/StatefulPartitionedCall,forward_simple_rnn_1/StatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
&forward_simple_rnn_1_while_cond_393494F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3H
Dforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393494___redundant_placeholder0^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393494___redundant_placeholder1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393494___redundant_placeholder2^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393494___redundant_placeholder3^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_393494___redundant_placeholder4'
#forward_simple_rnn_1_while_identity
?
forward_simple_rnn_1/while/LessLess&forward_simple_rnn_1_while_placeholderDforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_1/while/IdentityIdentity#forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390028

inputs

states0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?
?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_394833

inputs
states_00
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?4
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390104

inputs+
simple_rnn_cell_10_390029:@@'
simple_rnn_cell_10_390031:@+
simple_rnn_cell_10_390033:@@
identity??*simple_rnn_cell_10/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
*simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_10_390029simple_rnn_cell_10_390031simple_rnn_cell_10_390033*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390028n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_10_390029simple_rnn_cell_10_390031simple_rnn_cell_10_390033*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390041*
condR
while_cond_390040*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@{
NoOpNoOp+^simple_rnn_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2X
*simple_rnn_cell_10/StatefulPartitionedCall*simple_rnn_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_392657

inputs6
#embedding_1_embedding_lookup_392374:	?*@h
Vbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@e
Wbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@j
Xbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@i
Wbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@f
Xbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@k
Ybidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@9
&dense_2_matmul_readvariableop_resource:	?@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:
identity??Obidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Nbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?Pbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?+bidirectional_1/backward_simple_rnn_1/while?Nbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Mbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp?Obidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?*bidirectional_1/forward_simple_rnn_1/while?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupj
embedding_1/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_392374embedding_1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/392374*4
_output_shapes"
 :??????????????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/392374*4
_output_shapes"
 :??????????????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@[
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
*bidirectional_1/forward_simple_rnn_1/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
8bidirectional_1/forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:bidirectional_1/forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:bidirectional_1/forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2bidirectional_1/forward_simple_rnn_1/strided_sliceStridedSlice3bidirectional_1/forward_simple_rnn_1/Shape:output:0Abidirectional_1/forward_simple_rnn_1/strided_slice/stack:output:0Cbidirectional_1/forward_simple_rnn_1/strided_slice/stack_1:output:0Cbidirectional_1/forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3bidirectional_1/forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
1bidirectional_1/forward_simple_rnn_1/zeros/packedPack;bidirectional_1/forward_simple_rnn_1/strided_slice:output:0<bidirectional_1/forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:u
0bidirectional_1/forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
*bidirectional_1/forward_simple_rnn_1/zerosFill:bidirectional_1/forward_simple_rnn_1/zeros/packed:output:09bidirectional_1/forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@?
3bidirectional_1/forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
.bidirectional_1/forward_simple_rnn_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0<bidirectional_1/forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
,bidirectional_1/forward_simple_rnn_1/Shape_1Shape2bidirectional_1/forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:?
:bidirectional_1/forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4bidirectional_1/forward_simple_rnn_1/strided_slice_1StridedSlice5bidirectional_1/forward_simple_rnn_1/Shape_1:output:0Cbidirectional_1/forward_simple_rnn_1/strided_slice_1/stack:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_1:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
3bidirectional_1/forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/bidirectional_1/forward_simple_rnn_1/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0<bidirectional_1/forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :???????????????????
5bidirectional_1/forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
0bidirectional_1/forward_simple_rnn_1/transpose_1	Transpose8bidirectional_1/forward_simple_rnn_1/ExpandDims:output:0>bidirectional_1/forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :???????????????????
@bidirectional_1/forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2bidirectional_1/forward_simple_rnn_1/TensorArrayV2TensorListReserveIbidirectional_1/forward_simple_rnn_1/TensorArrayV2/element_shape:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Zbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Lbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor2bidirectional_1/forward_simple_rnn_1/transpose:y:0cbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
:bidirectional_1/forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4bidirectional_1/forward_simple_rnn_1/strided_slice_2StridedSlice2bidirectional_1/forward_simple_rnn_1/transpose:y:0Cbidirectional_1/forward_simple_rnn_1/strided_slice_2/stack:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_1:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Mbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpVbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul=bidirectional_1/forward_simple_rnn_1/strided_slice_2:output:0Ubidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Nbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpWbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
?bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAddHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Vbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Obidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpXbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
@bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul3bidirectional_1/forward_simple_rnn_1/zeros:output:0Wbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
;bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/addAddV2Hbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0Jbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh?bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
Bbidirectional_1/forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
4bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1TensorListReserveKbidirectional_1/forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???k
)bidirectional_1/forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : ?
Bbidirectional_1/forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4bidirectional_1/forward_simple_rnn_1/TensorArrayV2_2TensorListReserveKbidirectional_1/forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
\bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Nbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor4bidirectional_1/forward_simple_rnn_1/transpose_1:y:0ebidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
/bidirectional_1/forward_simple_rnn_1/zeros_like	ZerosLike@bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@?
=bidirectional_1/forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????y
7bidirectional_1/forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?

*bidirectional_1/forward_simple_rnn_1/whileWhile@bidirectional_1/forward_simple_rnn_1/while/loop_counter:output:0Fbidirectional_1/forward_simple_rnn_1/while/maximum_iterations:output:02bidirectional_1/forward_simple_rnn_1/time:output:0=bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1:handle:03bidirectional_1/forward_simple_rnn_1/zeros_like:y:03bidirectional_1/forward_simple_rnn_1/zeros:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0\bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0^bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Vbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceWbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceXbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *B
body:R8
6bidirectional_1_forward_simple_rnn_1_while_body_392429*B
cond:R8
6bidirectional_1_forward_simple_rnn_1_while_cond_392428*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Ubidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Gbidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack3bidirectional_1/forward_simple_rnn_1/while:output:3^bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0?
:bidirectional_1/forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
<bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4bidirectional_1/forward_simple_rnn_1/strided_slice_3StridedSlicePbidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Cbidirectional_1/forward_simple_rnn_1/strided_slice_3/stack:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_1:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
5bidirectional_1/forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
0bidirectional_1/forward_simple_rnn_1/transpose_2	TransposePbidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0>bidirectional_1/forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
+bidirectional_1/backward_simple_rnn_1/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
9bidirectional_1/backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;bidirectional_1/backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;bidirectional_1/backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3bidirectional_1/backward_simple_rnn_1/strided_sliceStridedSlice4bidirectional_1/backward_simple_rnn_1/Shape:output:0Bbidirectional_1/backward_simple_rnn_1/strided_slice/stack:output:0Dbidirectional_1/backward_simple_rnn_1/strided_slice/stack_1:output:0Dbidirectional_1/backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4bidirectional_1/backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
2bidirectional_1/backward_simple_rnn_1/zeros/packedPack<bidirectional_1/backward_simple_rnn_1/strided_slice:output:0=bidirectional_1/backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1bidirectional_1/backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+bidirectional_1/backward_simple_rnn_1/zerosFill;bidirectional_1/backward_simple_rnn_1/zeros/packed:output:0:bidirectional_1/backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@?
4bidirectional_1/backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
/bidirectional_1/backward_simple_rnn_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0=bidirectional_1/backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
-bidirectional_1/backward_simple_rnn_1/Shape_1Shape3bidirectional_1/backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:?
;bidirectional_1/backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5bidirectional_1/backward_simple_rnn_1/strided_slice_1StridedSlice6bidirectional_1/backward_simple_rnn_1/Shape_1:output:0Dbidirectional_1/backward_simple_rnn_1/strided_slice_1/stack:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_1:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
4bidirectional_1/backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0bidirectional_1/backward_simple_rnn_1/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0=bidirectional_1/backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :???????????????????
6bidirectional_1/backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1bidirectional_1/backward_simple_rnn_1/transpose_1	Transpose9bidirectional_1/backward_simple_rnn_1/ExpandDims:output:0?bidirectional_1/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :???????????????????
Abidirectional_1/backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
3bidirectional_1/backward_simple_rnn_1/TensorArrayV2TensorListReserveJbidirectional_1/backward_simple_rnn_1/TensorArrayV2/element_shape:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???~
4bidirectional_1/backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
/bidirectional_1/backward_simple_rnn_1/ReverseV2	ReverseV23bidirectional_1/backward_simple_rnn_1/transpose:y:0=bidirectional_1/backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
[bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Mbidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor8bidirectional_1/backward_simple_rnn_1/ReverseV2:output:0dbidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
;bidirectional_1/backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5bidirectional_1/backward_simple_rnn_1/strided_slice_2StridedSlice3bidirectional_1/backward_simple_rnn_1/transpose:y:0Dbidirectional_1/backward_simple_rnn_1/strided_slice_2/stack:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_1:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Nbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpWbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul>bidirectional_1/backward_simple_rnn_1/strided_slice_2:output:0Vbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Obidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpXbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
@bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAddIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Wbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Pbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpYbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Abidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul4bidirectional_1/backward_simple_rnn_1/zeros:output:0Xbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
<bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/addAddV2Ibidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0Kbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh@bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
Cbidirectional_1/backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
5bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1TensorListReserveLbidirectional_1/backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
*bidirectional_1/backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : ?
6bidirectional_1/backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
1bidirectional_1/backward_simple_rnn_1/ReverseV2_1	ReverseV25bidirectional_1/backward_simple_rnn_1/transpose_1:y:0?bidirectional_1/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :???????????????????
Cbidirectional_1/backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5bidirectional_1/backward_simple_rnn_1/TensorArrayV2_2TensorListReserveLbidirectional_1/backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
]bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Obidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor:bidirectional_1/backward_simple_rnn_1/ReverseV2_1:output:0fbidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
0bidirectional_1/backward_simple_rnn_1/zeros_like	ZerosLikeAbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@?
>bidirectional_1/backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????z
8bidirectional_1/backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
+bidirectional_1/backward_simple_rnn_1/whileWhileAbidirectional_1/backward_simple_rnn_1/while/loop_counter:output:0Gbidirectional_1/backward_simple_rnn_1/while/maximum_iterations:output:03bidirectional_1/backward_simple_rnn_1/time:output:0>bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1:handle:04bidirectional_1/backward_simple_rnn_1/zeros_like:y:04bidirectional_1/backward_simple_rnn_1/zeros:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0]bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0_bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Wbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceXbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceYbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *C
body;R9
7bidirectional_1_backward_simple_rnn_1_while_body_392561*C
cond;R9
7bidirectional_1_backward_simple_rnn_1_while_cond_392560*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Vbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Hbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack4bidirectional_1/backward_simple_rnn_1/while:output:3_bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0?
;bidirectional_1/backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
=bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5bidirectional_1/backward_simple_rnn_1/strided_slice_3StridedSliceQbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Dbidirectional_1/backward_simple_rnn_1/strided_slice_3/stack:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_1:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
6bidirectional_1/backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1bidirectional_1/backward_simple_rnn_1/transpose_2	TransposeQbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional_1/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@]
bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
bidirectional_1/concatConcatV2=bidirectional_1/forward_simple_rnn_1/strided_slice_3:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_3:output:0$bidirectional_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_2/MatMulMatMulbidirectional_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpP^bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpO^bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpQ^bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp,^bidirectional_1/backward_simple_rnn_1/whileO^bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpN^bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpP^bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp+^bidirectional_1/forward_simple_rnn_1/while^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2?
Obidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpObidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Nbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpNbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Pbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpPbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2Z
+bidirectional_1/backward_simple_rnn_1/while+bidirectional_1/backward_simple_rnn_1/while2?
Nbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpNbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Mbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpMbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Obidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpObidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp2X
*bidirectional_1/forward_simple_rnn_1/while*bidirectional_1/forward_simple_rnn_1/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
Csequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751?
|sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_loop_counter?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_maximum_iterationsG
Csequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholderI
Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_1I
Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_2I
Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_3?
~sequential_1_bidirectional_1_forward_simple_rnn_1_while_less_sequential_1_bidirectional_1_forward_simple_rnn_1_strided_slice_1?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751___redundant_placeholder0?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751___redundant_placeholder1?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751___redundant_placeholder2?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751___redundant_placeholder3?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_cond_389751___redundant_placeholder4D
@sequential_1_bidirectional_1_forward_simple_rnn_1_while_identity
?
<sequential_1/bidirectional_1/forward_simple_rnn_1/while/LessLessCsequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder~sequential_1_bidirectional_1_forward_simple_rnn_1_while_less_sequential_1_bidirectional_1_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: ?
@sequential_1/bidirectional_1/forward_simple_rnn_1/while/IdentityIdentity@sequential_1/bidirectional_1/forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "?
@sequential_1_bidirectional_1_forward_simple_rnn_1_while_identityIsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_393730

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390148

inputs

states0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
??
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391413

inputs
mask
X
Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@U
Gforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@V
Hbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity???backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?backward_simple_rnn_1/while?>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp??forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/whileP
forward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"forward_simple_rnn_1/strided_sliceStridedSlice#forward_simple_rnn_1/Shape:output:01forward_simple_rnn_1/strided_slice/stack:output:03forward_simple_rnn_1/strided_slice/stack_1:output:03forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
!forward_simple_rnn_1/zeros/packedPack+forward_simple_rnn_1/strided_slice:output:0,forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
forward_simple_rnn_1/zerosFill*forward_simple_rnn_1/zeros/packed:output:0)forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@x
#forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
forward_simple_rnn_1/transpose	Transposeinputs,forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@n
forward_simple_rnn_1/Shape_1Shape"forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_1StridedSlice%forward_simple_rnn_1/Shape_1:output:03forward_simple_rnn_1/strided_slice_1/stack:output:05forward_simple_rnn_1/strided_slice_1/stack_1:output:05forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
forward_simple_rnn_1/ExpandDims
ExpandDimsmask,forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????z
%forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_1	Transpose(forward_simple_rnn_1/ExpandDims:output:0.forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????{
0forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"forward_simple_rnn_1/TensorArrayV2TensorListReserve9forward_simple_rnn_1/TensorArrayV2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Jforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
<forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_1/transpose:y:0Sforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???t
*forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_2StridedSlice"forward_simple_rnn_1/transpose:y:03forward_simple_rnn_1/strided_slice_2/stack:output:05forward_simple_rnn_1/strided_slice_2/stack_1:output:05forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
.forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul-forward_simple_rnn_1/strided_slice_2:output:0Eforward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAdd8forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Fforward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
0forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul#forward_simple_rnn_1/zeros:output:0Gforward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+forward_simple_rnn_1/simple_rnn_cell_10/addAddV28forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0:forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
,forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
$forward_simple_rnn_1/TensorArrayV2_1TensorListReserve;forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???[
forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
$forward_simple_rnn_1/TensorArrayV2_2TensorListReserve;forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Lforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
>forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor$forward_simple_rnn_1/transpose_1:y:0Uforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
forward_simple_rnn_1/zeros_like	ZerosLike0forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@x
-forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????i
'forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
forward_simple_rnn_1/whileWhile0forward_simple_rnn_1/while/loop_counter:output:06forward_simple_rnn_1/while/maximum_iterations:output:0"forward_simple_rnn_1/time:output:0-forward_simple_rnn_1/TensorArrayV2_1:handle:0#forward_simple_rnn_1/zeros_like:y:0#forward_simple_rnn_1/zeros:output:0-forward_simple_rnn_1/strided_slice_1:output:0Lforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Nforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&forward_simple_rnn_1_while_body_391198*2
cond*R(
&forward_simple_rnn_1_while_cond_391197*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Eforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
7forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_1/while:output:3Nforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0}
*forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
,forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_3StridedSlice@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_1/strided_slice_3/stack:output:05forward_simple_rnn_1/strided_slice_3/stack_1:output:05forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskz
%forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_2	Transpose@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@Q
backward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#backward_simple_rnn_1/strided_sliceStridedSlice$backward_simple_rnn_1/Shape:output:02backward_simple_rnn_1/strided_slice/stack:output:04backward_simple_rnn_1/strided_slice/stack_1:output:04backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"backward_simple_rnn_1/zeros/packedPack,backward_simple_rnn_1/strided_slice:output:0-backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
backward_simple_rnn_1/zerosFill+backward_simple_rnn_1/zeros/packed:output:0*backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@y
$backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
backward_simple_rnn_1/transpose	Transposeinputs-backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@p
backward_simple_rnn_1/Shape_1Shape#backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_1StridedSlice&backward_simple_rnn_1/Shape_1:output:04backward_simple_rnn_1/strided_slice_1/stack:output:06backward_simple_rnn_1/strided_slice_1/stack_1:output:06backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 backward_simple_rnn_1/ExpandDims
ExpandDimsmask-backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????{
&backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_1	Transpose)backward_simple_rnn_1/ExpandDims:output:0/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????|
1backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#backward_simple_rnn_1/TensorArrayV2TensorListReserve:backward_simple_rnn_1/TensorArrayV2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???n
$backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
backward_simple_rnn_1/ReverseV2	ReverseV2#backward_simple_rnn_1/transpose:y:0-backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
Kbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
=backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_1/ReverseV2:output:0Tbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_2StridedSlice#backward_simple_rnn_1/transpose:y:04backward_simple_rnn_1/strided_slice_2/stack:output:06backward_simple_rnn_1/strided_slice_2/stack_1:output:06backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul.backward_simple_rnn_1/strided_slice_2:output:0Fbackward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAdd9backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Gbackward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
1backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul$backward_simple_rnn_1/zeros:output:0Hbackward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,backward_simple_rnn_1/simple_rnn_cell_11/addAddV29backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0;backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
-backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh0backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%backward_simple_rnn_1/TensorArrayV2_1TensorListReserve<backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
&backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
!backward_simple_rnn_1/ReverseV2_1	ReverseV2%backward_simple_rnn_1/transpose_1:y:0/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :??????????????????~
3backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%backward_simple_rnn_1/TensorArrayV2_2TensorListReserve<backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Mbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
?backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor*backward_simple_rnn_1/ReverseV2_1:output:0Vbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
 backward_simple_rnn_1/zeros_like	ZerosLike1backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@y
.backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
backward_simple_rnn_1/whileWhile1backward_simple_rnn_1/while/loop_counter:output:07backward_simple_rnn_1/while/maximum_iterations:output:0#backward_simple_rnn_1/time:output:0.backward_simple_rnn_1/TensorArrayV2_1:handle:0$backward_simple_rnn_1/zeros_like:y:0$backward_simple_rnn_1/zeros:output:0.backward_simple_rnn_1/strided_slice_1:output:0Mbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Obackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'backward_simple_rnn_1_while_body_391330*3
cond+R)
'backward_simple_rnn_1_while_cond_391329*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Fbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_1/while:output:3Obackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0~
+backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_3StridedSliceAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_1/strided_slice_3/stack:output:06backward_simple_rnn_1/strided_slice_3/stack_1:output:06backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_2	TransposeAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2-forward_simple_rnn_1/strided_slice_3:output:0.backward_simple_rnn_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp@^backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?^backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpA^backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp^backward_simple_rnn_1/while?^forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>^forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp@^forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp^forward_simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????????????@:??????????????????: : : : : : 2?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2:
backward_simple_rnn_1/whilebackward_simple_rnn_1/while2?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp28
forward_simple_rnn_1/whileforward_simple_rnn_1/while:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?,
?
while_body_394051
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_10/MatMul/ReadVariableOp?0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_391994
embedding_1_input%
embedding_1_391965:	?*@(
bidirectional_1_391970:@@$
bidirectional_1_391972:@(
bidirectional_1_391974:@@(
bidirectional_1_391976:@@$
bidirectional_1_391978:@(
bidirectional_1_391980:@@!
dense_2_391983:	?@
dense_2_391985:@ 
dense_3_391988:@
dense_3_391990:
identity??'bidirectional_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_391965*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_391141[
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
embedding_1/NotEqualNotEqualembedding_1_inputembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0bidirectional_1_391970bidirectional_1_391972bidirectional_1_391974bidirectional_1_391976bidirectional_1_391978bidirectional_1_391980*
Tin

2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391794?
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_391983dense_2_391985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_391438?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_391988dense_3_391990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_391454w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^bidirectional_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
?V
?
'backward_simple_rnn_1_while_body_393360H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3G
Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0?
backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@^
Pbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_1_while_identity*
&backward_simple_rnn_1_while_identity_1*
&backward_simple_rnn_1_while_identity_2*
&backward_simple_rnn_1_while_identity_3*
&backward_simple_rnn_1_while_identity_4*
&backward_simple_rnn_1_while_identity_5E
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@\
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
Mbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
?backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderVbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Obackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Abackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderXbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
5backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulFbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
6backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd?backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0Mbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
7backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul)backward_simple_rnn_1_while_placeholder_3Nbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
2backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2?backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Abackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanh6backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@{
*backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
 backward_simple_rnn_1/while/TileTileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:03backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
$backward_simple_rnn_1/while/SelectV2SelectV2)backward_simple_rnn_1/while/Tile:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@}
,backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
"backward_simple_rnn_1/while/Tile_1TileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:05backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
&backward_simple_rnn_1/while/SelectV2_1SelectV2+backward_simple_rnn_1/while/Tile_1:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_1_while_placeholder_1'backward_simple_rnn_1_while_placeholder-backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???c
!backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
backward_simple_rnn_1/while/addAddV2'backward_simple_rnn_1_while_placeholder*backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!backward_simple_rnn_1/while/add_1AddV2Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counter,backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$backward_simple_rnn_1/while/IdentityIdentity%backward_simple_rnn_1/while/add_1:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_1IdentityJbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_2Identity#backward_simple_rnn_1/while/add:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_3IdentityPbackward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_4Identity-backward_simple_rnn_1/while/SelectV2:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
&backward_simple_rnn_1/while/Identity_5Identity/backward_simple_rnn_1/while/SelectV2_1:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
 backward_simple_rnn_1/while/NoOpNoOpF^backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpG^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0"U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0"Y
&backward_simple_rnn_1_while_identity_1/backward_simple_rnn_1/while/Identity_1:output:0"Y
&backward_simple_rnn_1_while_identity_2/backward_simple_rnn_1/while/Identity_2:output:0"Y
&backward_simple_rnn_1_while_identity_3/backward_simple_rnn_1/while/Identity_3:output:0"Y
&backward_simple_rnn_1_while_identity_4/backward_simple_rnn_1/while/Identity_4:output:0"Y
&backward_simple_rnn_1_while_identity_5/backward_simple_rnn_1/while/Identity_5:output:0"?
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcePbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourceObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpEbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpDbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpFbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?

6bidirectional_1_forward_simple_rnn_1_while_cond_392428f
bbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_loop_counterl
hbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations:
6bidirectional_1_forward_simple_rnn_1_while_placeholder<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_1<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_2<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_3h
dbidirectional_1_forward_simple_rnn_1_while_less_bidirectional_1_forward_simple_rnn_1_strided_slice_1~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392428___redundant_placeholder0~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392428___redundant_placeholder1~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392428___redundant_placeholder2~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392428___redundant_placeholder3~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392428___redundant_placeholder47
3bidirectional_1_forward_simple_rnn_1_while_identity
?
/bidirectional_1/forward_simple_rnn_1/while/LessLess6bidirectional_1_forward_simple_rnn_1_while_placeholderdbidirectional_1_forward_simple_rnn_1_while_less_bidirectional_1_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: ?
3bidirectional_1/forward_simple_rnn_1/while/IdentityIdentity3bidirectional_1/forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "s
3bidirectional_1_forward_simple_rnn_1_while_identity<bidirectional_1/forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?V
?
'backward_simple_rnn_1_while_body_393627H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3G
Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0?
backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@^
Pbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_1_while_identity*
&backward_simple_rnn_1_while_identity_1*
&backward_simple_rnn_1_while_identity_2*
&backward_simple_rnn_1_while_identity_3*
&backward_simple_rnn_1_while_identity_4*
&backward_simple_rnn_1_while_identity_5E
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@\
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
Mbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
?backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderVbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Obackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Abackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderXbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
5backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulFbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
6backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd?backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0Mbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
7backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul)backward_simple_rnn_1_while_placeholder_3Nbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
2backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2?backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Abackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanh6backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@{
*backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
 backward_simple_rnn_1/while/TileTileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:03backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
$backward_simple_rnn_1/while/SelectV2SelectV2)backward_simple_rnn_1/while/Tile:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@}
,backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
"backward_simple_rnn_1/while/Tile_1TileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:05backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
&backward_simple_rnn_1/while/SelectV2_1SelectV2+backward_simple_rnn_1/while/Tile_1:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_1_while_placeholder_1'backward_simple_rnn_1_while_placeholder-backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???c
!backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
backward_simple_rnn_1/while/addAddV2'backward_simple_rnn_1_while_placeholder*backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!backward_simple_rnn_1/while/add_1AddV2Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counter,backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$backward_simple_rnn_1/while/IdentityIdentity%backward_simple_rnn_1/while/add_1:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_1IdentityJbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_2Identity#backward_simple_rnn_1/while/add:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_3IdentityPbackward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_4Identity-backward_simple_rnn_1/while/SelectV2:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
&backward_simple_rnn_1/while/Identity_5Identity/backward_simple_rnn_1/while/SelectV2_1:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
 backward_simple_rnn_1/while/NoOpNoOpF^backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpG^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0"U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0"Y
&backward_simple_rnn_1_while_identity_1/backward_simple_rnn_1/while/Identity_1:output:0"Y
&backward_simple_rnn_1_while_identity_2/backward_simple_rnn_1/while/Identity_2:output:0"Y
&backward_simple_rnn_1_while_identity_3/backward_simple_rnn_1/while/Identity_3:output:0"Y
&backward_simple_rnn_1_while_identity_4/backward_simple_rnn_1/while/Identity_4:output:0"Y
&backward_simple_rnn_1_while_identity_5/backward_simple_rnn_1/while/Identity_5:output:0"?
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcePbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourceObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpEbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpDbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpFbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_393749

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?@
?
&forward_simple_rnn_1_while_body_392786F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2E
Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0?
}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@]
Oforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_1_while_identity)
%forward_simple_rnn_1_while_identity_1)
%forward_simple_rnn_1_while_identity_2)
%forward_simple_rnn_1_while_identity_3)
%forward_simple_rnn_1_while_identity_4C
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@[
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
Lforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
>forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderUforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
4forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulEforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
5forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd>forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0Lforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
6forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul(forward_simple_rnn_1_while_placeholder_2Mforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2>forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0@forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanh5forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_1_while_placeholder_1&forward_simple_rnn_1_while_placeholder6forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???b
 forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
forward_simple_rnn_1/while/addAddV2&forward_simple_rnn_1_while_placeholder)forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
 forward_simple_rnn_1/while/add_1AddV2Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counter+forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
#forward_simple_rnn_1/while/IdentityIdentity$forward_simple_rnn_1/while/add_1:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_1IdentityHforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_2Identity"forward_simple_rnn_1/while/add:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_3IdentityOforward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_4Identity6forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
forward_simple_rnn_1/while/NoOpNoOpE^forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpF^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0"S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0"W
%forward_simple_rnn_1_while_identity_1.forward_simple_rnn_1/while/Identity_1:output:0"W
%forward_simple_rnn_1_while_identity_2.forward_simple_rnn_1/while/Identity_2:output:0"W
%forward_simple_rnn_1_while_identity_3.forward_simple_rnn_1/while/Identity_3:output:0"W
%forward_simple_rnn_1_while_identity_4.forward_simple_rnn_1/while/Identity_4:output:0"?
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourceOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcePforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpDforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpCforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpEforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?,
?
while_body_390882
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_11/MatMul/ReadVariableOp?0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_11/TanhTanh while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_11/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?>
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_391078

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_10/BiasAdd/ReadVariableOp?(simple_rnn_cell_10/MatMul/ReadVariableOp?*simple_rnn_cell_10/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_391012*
condR
while_cond_391011*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_391962
embedding_1_input%
embedding_1_391933:	?*@(
bidirectional_1_391938:@@$
bidirectional_1_391940:@(
bidirectional_1_391942:@@(
bidirectional_1_391944:@@$
bidirectional_1_391946:@(
bidirectional_1_391948:@@!
dense_2_391951:	?@
dense_2_391953:@ 
dense_3_391956:@
dense_3_391958:
identity??'bidirectional_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_391933*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_391141[
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
embedding_1/NotEqualNotEqualembedding_1_inputembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0bidirectional_1_391938bidirectional_1_391940bidirectional_1_391942bidirectional_1_391944bidirectional_1_391946bidirectional_1_391948*
Tin

2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391413?
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_391951dense_2_391953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_391438?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_391956dense_3_391958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_391454w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^bidirectional_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
?

?
3__inference_simple_rnn_cell_10_layer_call_fn_394737

inputs
states_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390148o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?>
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394117

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_10/BiasAdd/ReadVariableOp?(simple_rnn_cell_10/MatMul/ReadVariableOp?*simple_rnn_cell_10/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_394051*
condR
while_cond_394050*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?A
?
'backward_simple_rnn_1_while_body_393108H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2G
Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0?
backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@^
Pbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_1_while_identity*
&backward_simple_rnn_1_while_identity_1*
&backward_simple_rnn_1_while_identity_2*
&backward_simple_rnn_1_while_identity_3*
&backward_simple_rnn_1_while_identity_4E
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@\
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
Mbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
?backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderVbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
5backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulFbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
6backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd?backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0Mbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
7backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul)backward_simple_rnn_1_while_placeholder_2Nbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
2backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2?backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Abackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanh6backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_1_while_placeholder_1'backward_simple_rnn_1_while_placeholder7backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???c
!backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
backward_simple_rnn_1/while/addAddV2'backward_simple_rnn_1_while_placeholder*backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!backward_simple_rnn_1/while/add_1AddV2Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counter,backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$backward_simple_rnn_1/while/IdentityIdentity%backward_simple_rnn_1/while/add_1:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_1IdentityJbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_2Identity#backward_simple_rnn_1/while/add:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_3IdentityPbackward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_4Identity7backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
 backward_simple_rnn_1/while/NoOpNoOpF^backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpG^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0"U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0"Y
&backward_simple_rnn_1_while_identity_1/backward_simple_rnn_1/while/Identity_1:output:0"Y
&backward_simple_rnn_1_while_identity_2/backward_simple_rnn_1/while/Identity_2:output:0"Y
&backward_simple_rnn_1_while_identity_3/backward_simple_rnn_1/while/Identity_3:output:0"Y
&backward_simple_rnn_1_while_identity_4/backward_simple_rnn_1/while/Identity_4:output:0"?
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcePbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourceObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpEbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpDbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpFbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?U
?
&forward_simple_rnn_1_while_body_391579F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3E
Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0?
}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@]
Oforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_1_while_identity)
%forward_simple_rnn_1_while_identity_1)
%forward_simple_rnn_1_while_identity_2)
%forward_simple_rnn_1_while_identity_3)
%forward_simple_rnn_1_while_identity_4)
%forward_simple_rnn_1_while_identity_5C
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor^
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@[
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
Lforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
>forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderUforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Nforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderWforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
4forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulEforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
5forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd>forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0Lforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
6forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul(forward_simple_rnn_1_while_placeholder_3Mforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2>forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0@forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanh5forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@z
)forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
forward_simple_rnn_1/while/TileTileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:02forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
#forward_simple_rnn_1/while/SelectV2SelectV2(forward_simple_rnn_1/while/Tile:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@|
+forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
!forward_simple_rnn_1/while/Tile_1TileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:04forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
%forward_simple_rnn_1/while/SelectV2_1SelectV2*forward_simple_rnn_1/while/Tile_1:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_1_while_placeholder_1&forward_simple_rnn_1_while_placeholder,forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???b
 forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
forward_simple_rnn_1/while/addAddV2&forward_simple_rnn_1_while_placeholder)forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
 forward_simple_rnn_1/while/add_1AddV2Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counter+forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
#forward_simple_rnn_1/while/IdentityIdentity$forward_simple_rnn_1/while/add_1:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_1IdentityHforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_2Identity"forward_simple_rnn_1/while/add:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_3IdentityOforward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_4Identity,forward_simple_rnn_1/while/SelectV2:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
%forward_simple_rnn_1/while/Identity_5Identity.forward_simple_rnn_1/while/SelectV2_1:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
forward_simple_rnn_1/while/NoOpNoOpE^forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpF^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0"S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0"W
%forward_simple_rnn_1_while_identity_1.forward_simple_rnn_1/while/Identity_1:output:0"W
%forward_simple_rnn_1_while_identity_2.forward_simple_rnn_1/while/Identity_2:output:0"W
%forward_simple_rnn_1_while_identity_3.forward_simple_rnn_1/while/Identity_3:output:0"W
%forward_simple_rnn_1_while_identity_4.forward_simple_rnn_1/while/Identity_4:output:0"W
%forward_simple_rnn_1_while_identity_5.forward_simple_rnn_1/while/Identity_5:output:0"?
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourceOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcePforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpDforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpCforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpEforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393443

inputs
mask
X
Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@U
Gforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@V
Hbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity???backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?backward_simple_rnn_1/while?>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp??forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/whileP
forward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"forward_simple_rnn_1/strided_sliceStridedSlice#forward_simple_rnn_1/Shape:output:01forward_simple_rnn_1/strided_slice/stack:output:03forward_simple_rnn_1/strided_slice/stack_1:output:03forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
!forward_simple_rnn_1/zeros/packedPack+forward_simple_rnn_1/strided_slice:output:0,forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
forward_simple_rnn_1/zerosFill*forward_simple_rnn_1/zeros/packed:output:0)forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@x
#forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
forward_simple_rnn_1/transpose	Transposeinputs,forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@n
forward_simple_rnn_1/Shape_1Shape"forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_1StridedSlice%forward_simple_rnn_1/Shape_1:output:03forward_simple_rnn_1/strided_slice_1/stack:output:05forward_simple_rnn_1/strided_slice_1/stack_1:output:05forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
forward_simple_rnn_1/ExpandDims
ExpandDimsmask,forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????z
%forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_1	Transpose(forward_simple_rnn_1/ExpandDims:output:0.forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????{
0forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"forward_simple_rnn_1/TensorArrayV2TensorListReserve9forward_simple_rnn_1/TensorArrayV2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Jforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
<forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_1/transpose:y:0Sforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???t
*forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_2StridedSlice"forward_simple_rnn_1/transpose:y:03forward_simple_rnn_1/strided_slice_2/stack:output:05forward_simple_rnn_1/strided_slice_2/stack_1:output:05forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
.forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul-forward_simple_rnn_1/strided_slice_2:output:0Eforward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAdd8forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Fforward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
0forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul#forward_simple_rnn_1/zeros:output:0Gforward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+forward_simple_rnn_1/simple_rnn_cell_10/addAddV28forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0:forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
,forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
$forward_simple_rnn_1/TensorArrayV2_1TensorListReserve;forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???[
forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
$forward_simple_rnn_1/TensorArrayV2_2TensorListReserve;forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Lforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
>forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor$forward_simple_rnn_1/transpose_1:y:0Uforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
forward_simple_rnn_1/zeros_like	ZerosLike0forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@x
-forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????i
'forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
forward_simple_rnn_1/whileWhile0forward_simple_rnn_1/while/loop_counter:output:06forward_simple_rnn_1/while/maximum_iterations:output:0"forward_simple_rnn_1/time:output:0-forward_simple_rnn_1/TensorArrayV2_1:handle:0#forward_simple_rnn_1/zeros_like:y:0#forward_simple_rnn_1/zeros:output:0-forward_simple_rnn_1/strided_slice_1:output:0Lforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Nforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&forward_simple_rnn_1_while_body_393228*2
cond*R(
&forward_simple_rnn_1_while_cond_393227*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Eforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
7forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_1/while:output:3Nforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0}
*forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
,forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_3StridedSlice@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_1/strided_slice_3/stack:output:05forward_simple_rnn_1/strided_slice_3/stack_1:output:05forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskz
%forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_2	Transpose@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@Q
backward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#backward_simple_rnn_1/strided_sliceStridedSlice$backward_simple_rnn_1/Shape:output:02backward_simple_rnn_1/strided_slice/stack:output:04backward_simple_rnn_1/strided_slice/stack_1:output:04backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"backward_simple_rnn_1/zeros/packedPack,backward_simple_rnn_1/strided_slice:output:0-backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
backward_simple_rnn_1/zerosFill+backward_simple_rnn_1/zeros/packed:output:0*backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@y
$backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
backward_simple_rnn_1/transpose	Transposeinputs-backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@p
backward_simple_rnn_1/Shape_1Shape#backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_1StridedSlice&backward_simple_rnn_1/Shape_1:output:04backward_simple_rnn_1/strided_slice_1/stack:output:06backward_simple_rnn_1/strided_slice_1/stack_1:output:06backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 backward_simple_rnn_1/ExpandDims
ExpandDimsmask-backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????{
&backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_1	Transpose)backward_simple_rnn_1/ExpandDims:output:0/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????|
1backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#backward_simple_rnn_1/TensorArrayV2TensorListReserve:backward_simple_rnn_1/TensorArrayV2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???n
$backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
backward_simple_rnn_1/ReverseV2	ReverseV2#backward_simple_rnn_1/transpose:y:0-backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
Kbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
=backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_1/ReverseV2:output:0Tbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_2StridedSlice#backward_simple_rnn_1/transpose:y:04backward_simple_rnn_1/strided_slice_2/stack:output:06backward_simple_rnn_1/strided_slice_2/stack_1:output:06backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul.backward_simple_rnn_1/strided_slice_2:output:0Fbackward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAdd9backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Gbackward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
1backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul$backward_simple_rnn_1/zeros:output:0Hbackward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,backward_simple_rnn_1/simple_rnn_cell_11/addAddV29backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0;backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
-backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh0backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%backward_simple_rnn_1/TensorArrayV2_1TensorListReserve<backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
&backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
!backward_simple_rnn_1/ReverseV2_1	ReverseV2%backward_simple_rnn_1/transpose_1:y:0/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :??????????????????~
3backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%backward_simple_rnn_1/TensorArrayV2_2TensorListReserve<backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Mbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
?backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor*backward_simple_rnn_1/ReverseV2_1:output:0Vbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
 backward_simple_rnn_1/zeros_like	ZerosLike1backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@y
.backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
backward_simple_rnn_1/whileWhile1backward_simple_rnn_1/while/loop_counter:output:07backward_simple_rnn_1/while/maximum_iterations:output:0#backward_simple_rnn_1/time:output:0.backward_simple_rnn_1/TensorArrayV2_1:handle:0$backward_simple_rnn_1/zeros_like:y:0$backward_simple_rnn_1/zeros:output:0.backward_simple_rnn_1/strided_slice_1:output:0Mbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Obackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'backward_simple_rnn_1_while_body_393360*3
cond+R)
'backward_simple_rnn_1_while_cond_393359*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Fbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_1/while:output:3Obackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0~
+backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_3StridedSliceAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_1/strided_slice_3/stack:output:06backward_simple_rnn_1/strided_slice_3/stack_1:output:06backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_2	TransposeAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2-forward_simple_rnn_1/strided_slice_3:output:0.backward_simple_rnn_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp@^backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?^backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpA^backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp^backward_simple_rnn_1/while?^forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>^forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp@^forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp^forward_simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????????????@:??????????????????: : : : : : 2?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2:
backward_simple_rnn_1/whilebackward_simple_rnn_1/while2?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp28
forward_simple_rnn_1/whileforward_simple_rnn_1/while:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?

?
$__inference_signature_wrapper_392029
embedding_1_input
unknown:	?*@
	unknown_0:@@
	unknown_1:@
	unknown_2:@@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:	?@
	unknown_7:@
	unknown_8:@
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_389980o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
?,
?
while_body_394423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_11/MatMul/ReadVariableOp?0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_11/TanhTanh while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_11/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?,
?
while_body_394533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_11/MatMul/ReadVariableOp?0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_11/TanhTanh while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_11/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_sequential_1_layer_call_fn_392056

inputs
unknown:	?*@
	unknown_0:@@
	unknown_1:@
	unknown_2:@@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:	?@
	unknown_7:@
	unknown_8:@
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_391461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
while_cond_394158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_394158___redundant_placeholder04
0while_while_cond_394158___redundant_placeholder14
0while_while_cond_394158___redundant_placeholder24
0while_while_cond_394158___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?,
?
while_body_390733
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_11/MatMul/ReadVariableOp?0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_11/TanhTanh while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_11/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?V
?
'backward_simple_rnn_1_while_body_391330H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3G
Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0?
backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@^
Pbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_1_while_identity*
&backward_simple_rnn_1_while_identity_1*
&backward_simple_rnn_1_while_identity_2*
&backward_simple_rnn_1_while_identity_3*
&backward_simple_rnn_1_while_identity_4*
&backward_simple_rnn_1_while_identity_5E
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@\
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
Mbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
?backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderVbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Obackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Abackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderXbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
5backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulFbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
6backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd?backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0Mbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
7backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul)backward_simple_rnn_1_while_placeholder_3Nbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
2backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2?backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Abackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanh6backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@{
*backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
 backward_simple_rnn_1/while/TileTileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:03backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
$backward_simple_rnn_1/while/SelectV2SelectV2)backward_simple_rnn_1/while/Tile:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@}
,backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
"backward_simple_rnn_1/while/Tile_1TileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:05backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
&backward_simple_rnn_1/while/SelectV2_1SelectV2+backward_simple_rnn_1/while/Tile_1:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_1_while_placeholder_1'backward_simple_rnn_1_while_placeholder-backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???c
!backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
backward_simple_rnn_1/while/addAddV2'backward_simple_rnn_1_while_placeholder*backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!backward_simple_rnn_1/while/add_1AddV2Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counter,backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$backward_simple_rnn_1/while/IdentityIdentity%backward_simple_rnn_1/while/add_1:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_1IdentityJbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_2Identity#backward_simple_rnn_1/while/add:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_3IdentityPbackward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_4Identity-backward_simple_rnn_1/while/SelectV2:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
&backward_simple_rnn_1/while/Identity_5Identity/backward_simple_rnn_1/while/SelectV2_1:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
 backward_simple_rnn_1/while/NoOpNoOpF^backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpG^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0"U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0"Y
&backward_simple_rnn_1_while_identity_1/backward_simple_rnn_1/while/Identity_1:output:0"Y
&backward_simple_rnn_1_while_identity_2/backward_simple_rnn_1/while/Identity_2:output:0"Y
&backward_simple_rnn_1_while_identity_3/backward_simple_rnn_1/while/Identity_3:output:0"Y
&backward_simple_rnn_1_while_identity_4/backward_simple_rnn_1/while/Identity_4:output:0"Y
&backward_simple_rnn_1_while_identity_5/backward_simple_rnn_1/while/Identity_5:output:0"?
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcePbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourceObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpEbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpDbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpFbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?!
?
while_body_390335
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_11_390357_0:@@/
!while_simple_rnn_cell_11_390359_0:@3
!while_simple_rnn_cell_11_390361_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_11_390357:@@-
while_simple_rnn_cell_11_390359:@1
while_simple_rnn_cell_11_390361:@@??0while/simple_rnn_cell_11/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
0while/simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_11_390357_0!while_simple_rnn_cell_11_390359_0!while_simple_rnn_cell_11_390361_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390322?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_11/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity9while/simple_rnn_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????@

while/NoOpNoOp1^while/simple_rnn_cell_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_11_390357!while_simple_rnn_cell_11_390357_0"D
while_simple_rnn_cell_11_390359!while_simple_rnn_cell_11_390359_0"D
while_simple_rnn_cell_11_390361!while_simple_rnn_cell_11_390361_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2d
0while/simple_rnn_cell_11/StatefulPartitionedCall0while/simple_rnn_cell_11/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?5
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390398

inputs+
simple_rnn_cell_11_390323:@@'
simple_rnn_cell_11_390325:@+
simple_rnn_cell_11_390327:@@
identity??*simple_rnn_cell_11/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
*simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_11_390323simple_rnn_cell_11_390325simple_rnn_cell_11_390327*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390322n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_11_390323simple_rnn_cell_11_390325simple_rnn_cell_11_390327*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390335*
condR
while_cond_390334*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@{
NoOpNoOp+^simple_rnn_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2X
*simple_rnn_cell_11/StatefulPartitionedCall*simple_rnn_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?>
?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390682

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_10/BiasAdd/ReadVariableOp?(simple_rnn_cell_10/MatMul/ReadVariableOp?*simple_rnn_cell_10/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390616*
condR
while_cond_390615*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_392960
inputs_0X
Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@U
Gforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@V
Hbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity???backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?backward_simple_rnn_1/while?>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp??forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/whileR
forward_simple_rnn_1/ShapeShapeinputs_0*
T0*
_output_shapes
:r
(forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"forward_simple_rnn_1/strided_sliceStridedSlice#forward_simple_rnn_1/Shape:output:01forward_simple_rnn_1/strided_slice/stack:output:03forward_simple_rnn_1/strided_slice/stack_1:output:03forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
!forward_simple_rnn_1/zeros/packedPack+forward_simple_rnn_1/strided_slice:output:0,forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
forward_simple_rnn_1/zerosFill*forward_simple_rnn_1/zeros/packed:output:0)forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@x
#forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
forward_simple_rnn_1/transpose	Transposeinputs_0,forward_simple_rnn_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????n
forward_simple_rnn_1/Shape_1Shape"forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_1StridedSlice%forward_simple_rnn_1/Shape_1:output:03forward_simple_rnn_1/strided_slice_1/stack:output:05forward_simple_rnn_1/strided_slice_1/stack_1:output:05forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"forward_simple_rnn_1/TensorArrayV2TensorListReserve9forward_simple_rnn_1/TensorArrayV2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Jforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
<forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_1/transpose:y:0Sforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???t
*forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_2StridedSlice"forward_simple_rnn_1/transpose:y:03forward_simple_rnn_1/strided_slice_2/stack:output:05forward_simple_rnn_1/strided_slice_2/stack_1:output:05forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
.forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul-forward_simple_rnn_1/strided_slice_2:output:0Eforward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAdd8forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Fforward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
0forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul#forward_simple_rnn_1/zeros:output:0Gforward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+forward_simple_rnn_1/simple_rnn_cell_10/addAddV28forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0:forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
,forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
$forward_simple_rnn_1/TensorArrayV2_1TensorListReserve;forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???[
forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????i
'forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
forward_simple_rnn_1/whileWhile0forward_simple_rnn_1/while/loop_counter:output:06forward_simple_rnn_1/while/maximum_iterations:output:0"forward_simple_rnn_1/time:output:0-forward_simple_rnn_1/TensorArrayV2_1:handle:0#forward_simple_rnn_1/zeros:output:0-forward_simple_rnn_1/strided_slice_1:output:0Lforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&forward_simple_rnn_1_while_body_392786*2
cond*R(
&forward_simple_rnn_1_while_cond_392785*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
Eforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
7forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_1/while:output:3Nforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0}
*forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
,forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_3StridedSlice@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_1/strided_slice_3/stack:output:05forward_simple_rnn_1/strided_slice_3/stack_1:output:05forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskz
%forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_1	Transpose@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@S
backward_simple_rnn_1/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#backward_simple_rnn_1/strided_sliceStridedSlice$backward_simple_rnn_1/Shape:output:02backward_simple_rnn_1/strided_slice/stack:output:04backward_simple_rnn_1/strided_slice/stack_1:output:04backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"backward_simple_rnn_1/zeros/packedPack,backward_simple_rnn_1/strided_slice:output:0-backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
backward_simple_rnn_1/zerosFill+backward_simple_rnn_1/zeros/packed:output:0*backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@y
$backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
backward_simple_rnn_1/transpose	Transposeinputs_0-backward_simple_rnn_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????p
backward_simple_rnn_1/Shape_1Shape#backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_1StridedSlice&backward_simple_rnn_1/Shape_1:output:04backward_simple_rnn_1/strided_slice_1/stack:output:06backward_simple_rnn_1/strided_slice_1/stack_1:output:06backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#backward_simple_rnn_1/TensorArrayV2TensorListReserve:backward_simple_rnn_1/TensorArrayV2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???n
$backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
backward_simple_rnn_1/ReverseV2	ReverseV2#backward_simple_rnn_1/transpose:y:0-backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Kbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
=backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_1/ReverseV2:output:0Tbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_2StridedSlice#backward_simple_rnn_1/transpose:y:04backward_simple_rnn_1/strided_slice_2/stack:output:06backward_simple_rnn_1/strided_slice_2/stack_1:output:06backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul.backward_simple_rnn_1/strided_slice_2:output:0Fbackward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAdd9backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Gbackward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
1backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul$backward_simple_rnn_1/zeros:output:0Hbackward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,backward_simple_rnn_1/simple_rnn_cell_11/addAddV29backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0;backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
-backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh0backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%backward_simple_rnn_1/TensorArrayV2_1TensorListReserve<backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
backward_simple_rnn_1/whileWhile1backward_simple_rnn_1/while/loop_counter:output:07backward_simple_rnn_1/while/maximum_iterations:output:0#backward_simple_rnn_1/time:output:0.backward_simple_rnn_1/TensorArrayV2_1:handle:0$backward_simple_rnn_1/zeros:output:0.backward_simple_rnn_1/strided_slice_1:output:0Mbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'backward_simple_rnn_1_while_body_392892*3
cond+R)
'backward_simple_rnn_1_while_cond_392891*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
Fbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_1/while:output:3Obackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0~
+backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_3StridedSliceAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_1/strided_slice_3/stack:output:06backward_simple_rnn_1/strided_slice_3/stack_1:output:06backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_1	TransposeAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2-forward_simple_rnn_1/strided_slice_3:output:0.backward_simple_rnn_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp@^backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?^backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpA^backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp^backward_simple_rnn_1/while?^forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>^forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp@^forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp^forward_simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2:
backward_simple_rnn_1/whilebackward_simple_rnn_1/while2?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp28
forward_simple_rnn_1/whileforward_simple_rnn_1/while:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
??
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394379
inputs_0C
1simple_rnn_cell_11_matmul_readvariableop_resource:@@@
2simple_rnn_cell_11_biasadd_readvariableop_resource:@E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_11/BiasAdd/ReadVariableOp?(simple_rnn_cell_11/MatMul/ReadVariableOp?*simple_rnn_cell_11/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_11/TanhTanhsimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_394313*
condR
while_cond_394312*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
6__inference_backward_simple_rnn_1_layer_call_fn_394258

inputs
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390799o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
while_cond_390040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390040___redundant_placeholder04
0while_while_cond_390040___redundant_placeholder14
0while_while_cond_390040___redundant_placeholder24
0while_while_cond_390040___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_394771

inputs
states_00
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_391461

inputs%
embedding_1_391142:	?*@(
bidirectional_1_391414:@@$
bidirectional_1_391416:@(
bidirectional_1_391418:@@(
bidirectional_1_391420:@@$
bidirectional_1_391422:@(
bidirectional_1_391424:@@!
dense_2_391439:	?@
dense_2_391441:@ 
dense_3_391455:@
dense_3_391457:
identity??'bidirectional_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_391142*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_391141[
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0bidirectional_1_391414bidirectional_1_391416bidirectional_1_391418bidirectional_1_391420bidirectional_1_391422bidirectional_1_391424*
Tin

2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391413?
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_391439dense_2_391441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_391438?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_391455dense_3_391457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_391454w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^bidirectional_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?

6bidirectional_1_forward_simple_rnn_1_while_cond_392141f
bbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_loop_counterl
hbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations:
6bidirectional_1_forward_simple_rnn_1_while_placeholder<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_1<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_2<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_3h
dbidirectional_1_forward_simple_rnn_1_while_less_bidirectional_1_forward_simple_rnn_1_strided_slice_1~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392141___redundant_placeholder0~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392141___redundant_placeholder1~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392141___redundant_placeholder2~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392141___redundant_placeholder3~
zbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_cond_392141___redundant_placeholder47
3bidirectional_1_forward_simple_rnn_1_while_identity
?
/bidirectional_1/forward_simple_rnn_1/while/LessLess6bidirectional_1_forward_simple_rnn_1_while_placeholderdbidirectional_1_forward_simple_rnn_1_while_less_bidirectional_1_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: ?
3bidirectional_1/forward_simple_rnn_1/while/IdentityIdentity3bidirectional_1/forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "s
3bidirectional_1_forward_simple_rnn_1_while_identity<bidirectional_1/forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?
?
5__inference_forward_simple_rnn_1_layer_call_fn_393782

inputs
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?z
?
Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_body_389884?
~sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_loop_counter?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_maximum_iterationsH
Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholderJ
Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_1J
Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_2J
Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_3?
}sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_strided_slice_1_0?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0~
lsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@{
msequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@?
nsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@E
Asequential_1_bidirectional_1_backward_simple_rnn_1_while_identityG
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_1G
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_2G
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_3G
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_4G
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_5
{sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_strided_slice_1?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor|
jsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@y
ksequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@~
lsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??bsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?asequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?csequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
jsequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
\sequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholderssequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
lsequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
^sequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholderusequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
asequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOplsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Rsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulcsequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0isequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
bsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpmsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
Ssequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd\sequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0jsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
csequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpnsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Tsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMulFsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_3ksequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Osequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2\sequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0^sequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Psequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanhSsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
Gsequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
=sequential_1/bidirectional_1/backward_simple_rnn_1/while/TileTileesequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Psequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
Asequential_1/bidirectional_1/backward_simple_rnn_1/while/SelectV2SelectV2Fsequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile:output:0Tsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@?
Isequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
?sequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile_1Tileesequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Rsequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
Csequential_1/bidirectional_1/backward_simple_rnn_1/while/SelectV2_1SelectV2Hsequential_1/bidirectional_1/backward_simple_rnn_1/while/Tile_1:output:0Tsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0Fsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
]sequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholder_1Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholderJsequential_1/bidirectional_1/backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:????
>sequential_1/bidirectional_1/backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
<sequential_1/bidirectional_1/backward_simple_rnn_1/while/addAddV2Dsequential_1_bidirectional_1_backward_simple_rnn_1_while_placeholderGsequential_1/bidirectional_1/backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: ?
@sequential_1/bidirectional_1/backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
>sequential_1/bidirectional_1/backward_simple_rnn_1/while/add_1AddV2~sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_loop_counterIsequential_1/bidirectional_1/backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
Asequential_1/bidirectional_1/backward_simple_rnn_1/while/IdentityIdentityBsequential_1/bidirectional_1/backward_simple_rnn_1/while/add_1:z:0>^sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Csequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_1Identity?sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations>^sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Csequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_2Identity@sequential_1/bidirectional_1/backward_simple_rnn_1/while/add:z:0>^sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Csequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_3Identitymsequential_1/bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Csequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_4IdentityJsequential_1/bidirectional_1/backward_simple_rnn_1/while/SelectV2:output:0>^sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
Csequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_5IdentityLsequential_1/bidirectional_1/backward_simple_rnn_1/while/SelectV2_1:output:0>^sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
=sequential_1/bidirectional_1/backward_simple_rnn_1/while/NoOpNoOpc^sequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpb^sequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpd^sequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Asequential_1_bidirectional_1_backward_simple_rnn_1_while_identityJsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity:output:0"?
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_1Lsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_1:output:0"?
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_2Lsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_2:output:0"?
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_3Lsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_3:output:0"?
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_4Lsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_4:output:0"?
Csequential_1_bidirectional_1_backward_simple_rnn_1_while_identity_5Lsequential_1/bidirectional_1/backward_simple_rnn_1/while/Identity_5:output:0"?
{sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_strided_slice_1}sequential_1_bidirectional_1_backward_simple_rnn_1_while_sequential_1_bidirectional_1_backward_simple_rnn_1_strided_slice_1_0"?
ksequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcemsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
lsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourcensequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
jsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourcelsequential_1_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?sequential_1_bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
bsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpbsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
asequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpasequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
csequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpcsequential_1/bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?,
?
while_body_393943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_10/MatMul/ReadVariableOp?0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_3_layer_call_fn_393739

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_391454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
0__inference_bidirectional_1_layer_call_fn_392691
inputs_0
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_390810p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?!
?
while_body_390200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_10_390222_0:@@/
!while_simple_rnn_cell_10_390224_0:@3
!while_simple_rnn_cell_10_390226_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_10_390222:@@-
while_simple_rnn_cell_10_390224:@1
while_simple_rnn_cell_10_390226:@@??0while/simple_rnn_cell_10/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
0while/simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_10_390222_0!while_simple_rnn_cell_10_390224_0!while_simple_rnn_cell_10_390226_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390148?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity9while/simple_rnn_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????@

while/NoOpNoOp1^while/simple_rnn_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_10_390222!while_simple_rnn_cell_10_390222_0"D
while_simple_rnn_cell_10_390224!while_simple_rnn_cell_10_390224_0"D
while_simple_rnn_cell_10_390226!while_simple_rnn_cell_10_390226_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2d
0while/simple_rnn_cell_10/StatefulPartitionedCall0while/simple_rnn_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?@
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394709

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource:@@@
2simple_rnn_cell_11_biasadd_readvariableop_resource:@E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_11/BiasAdd/ReadVariableOp?(simple_rnn_cell_11/MatMul/ReadVariableOp?*simple_rnn_cell_11/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'????????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_11/TanhTanhsimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_394643*
condR
while_cond_394642*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?b
?
__inference__traced_save_394982
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop]
Ysavev2_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_read_readvariableopg
csavev2_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_read_readvariableop[
Wsavev2_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_read_readvariableop^
Zsavev2_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_read_readvariableoph
dsavev2_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_read_readvariableop\
Xsavev2_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableopd
`savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_m_read_readvariableopn
jsavev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_m_read_readvariableopb
^savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_m_read_readvariableope
asavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_m_read_readvariableopo
ksavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_m_read_readvariableopc
_savev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableopd
`savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_v_read_readvariableopn
jsavev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_v_read_readvariableopb
^savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_v_read_readvariableope
asavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_v_read_readvariableopo
ksavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_v_read_readvariableopc
_savev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*?
value?B?+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopYsavev2_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_read_readvariableopcsavev2_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_read_readvariableopWsavev2_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_read_readvariableopZsavev2_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_read_readvariableopdsavev2_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_read_readvariableopXsavev2_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop`savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_m_read_readvariableopjsavev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_m_read_readvariableop^savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_m_read_readvariableopasavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_m_read_readvariableopksavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_m_read_readvariableop_savev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_m_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop`savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_v_read_readvariableopjsavev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_v_read_readvariableop^savev2_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_v_read_readvariableopasavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_v_read_readvariableopksavev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_v_read_readvariableop_savev2_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?*@:	?@:@:@:: : : : : :@@:@@:@:@@:@@:@: : : : :	?*@:	?@:@:@::@@:@@:@:@@:@@:@:	?*@:	?@:@:@::@@:@@:@:@@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?*@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?*@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@:% !

_output_shapes
:	?*@:%!!

_output_shapes
:	?@: "

_output_shapes
:@:$# 

_output_shapes

:@: $

_output_shapes
::$% 

_output_shapes

:@@:$& 

_output_shapes

:@@: '

_output_shapes
:@:$( 

_output_shapes

:@@:$) 

_output_shapes

:@@: *

_output_shapes
:@:+

_output_shapes
: 
?U
?
&forward_simple_rnn_1_while_body_393228F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3E
Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0?
}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@]
Oforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_1_while_identity)
%forward_simple_rnn_1_while_identity_1)
%forward_simple_rnn_1_while_identity_2)
%forward_simple_rnn_1_while_identity_3)
%forward_simple_rnn_1_while_identity_4)
%forward_simple_rnn_1_while_identity_5C
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor^
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@[
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
Lforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
>forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderUforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Nforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderWforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
4forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulEforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
5forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd>forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0Lforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
6forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul(forward_simple_rnn_1_while_placeholder_3Mforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2>forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0@forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanh5forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@z
)forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
forward_simple_rnn_1/while/TileTileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:02forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
#forward_simple_rnn_1/while/SelectV2SelectV2(forward_simple_rnn_1/while/Tile:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@|
+forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
!forward_simple_rnn_1/while/Tile_1TileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:04forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
%forward_simple_rnn_1/while/SelectV2_1SelectV2*forward_simple_rnn_1/while/Tile_1:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_1_while_placeholder_1&forward_simple_rnn_1_while_placeholder,forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???b
 forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
forward_simple_rnn_1/while/addAddV2&forward_simple_rnn_1_while_placeholder)forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
 forward_simple_rnn_1/while/add_1AddV2Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counter+forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
#forward_simple_rnn_1/while/IdentityIdentity$forward_simple_rnn_1/while/add_1:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_1IdentityHforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_2Identity"forward_simple_rnn_1/while/add:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_3IdentityOforward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_4Identity,forward_simple_rnn_1/while/SelectV2:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
%forward_simple_rnn_1/while/Identity_5Identity.forward_simple_rnn_1/while/SelectV2_1:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
forward_simple_rnn_1/while/NoOpNoOpE^forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpF^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0"S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0"W
%forward_simple_rnn_1_while_identity_1.forward_simple_rnn_1/while/Identity_1:output:0"W
%forward_simple_rnn_1_while_identity_2.forward_simple_rnn_1/while/Identity_2:output:0"W
%forward_simple_rnn_1_while_identity_3.forward_simple_rnn_1/while/Identity_3:output:0"W
%forward_simple_rnn_1_while_identity_4.forward_simple_rnn_1/while/Identity_4:output:0"W
%forward_simple_rnn_1_while_identity_5.forward_simple_rnn_1/while/Identity_5:output:0"?
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourceOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcePforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpDforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpCforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpEforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
? 
"__inference__traced_restore_395118
file_prefix:
'assignvariableop_embedding_1_embeddings:	?*@4
!assignvariableop_1_dense_2_kernel:	?@-
assignvariableop_2_dense_2_bias:@3
!assignvariableop_3_dense_3_kernel:@-
assignvariableop_4_dense_3_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: d
Rassignvariableop_10_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel:@@n
\assignvariableop_11_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel:@@^
Passignvariableop_12_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias:@e
Sassignvariableop_13_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel:@@o
]assignvariableop_14_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel:@@_
Qassignvariableop_15_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias:@#
assignvariableop_16_total: #
assignvariableop_17_count: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: D
1assignvariableop_20_adam_embedding_1_embeddings_m:	?*@<
)assignvariableop_21_adam_dense_2_kernel_m:	?@5
'assignvariableop_22_adam_dense_2_bias_m:@;
)assignvariableop_23_adam_dense_3_kernel_m:@5
'assignvariableop_24_adam_dense_3_bias_m:k
Yassignvariableop_25_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_m:@@u
cassignvariableop_26_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_m:@@e
Wassignvariableop_27_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_m:@l
Zassignvariableop_28_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_m:@@v
dassignvariableop_29_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_m:@@f
Xassignvariableop_30_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_m:@D
1assignvariableop_31_adam_embedding_1_embeddings_v:	?*@<
)assignvariableop_32_adam_dense_2_kernel_v:	?@5
'assignvariableop_33_adam_dense_2_bias_v:@;
)assignvariableop_34_adam_dense_3_kernel_v:@5
'assignvariableop_35_adam_dense_3_bias_v:k
Yassignvariableop_36_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_v:@@u
cassignvariableop_37_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_v:@@e
Wassignvariableop_38_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_v:@l
Zassignvariableop_39_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_v:@@v
dassignvariableop_40_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_v:@@f
Xassignvariableop_41_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_v:@
identity_43??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*?
value?B?+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpRassignvariableop_10_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp\assignvariableop_11_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpPassignvariableop_12_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpSassignvariableop_13_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp]assignvariableop_14_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpQassignvariableop_15_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_embedding_1_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpYassignvariableop_25_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpcassignvariableop_26_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpWassignvariableop_27_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpZassignvariableop_28_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpdassignvariableop_29_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpXassignvariableop_30_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_embedding_1_embeddings_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_2_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_2_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_3_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_3_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpYassignvariableop_36_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpcassignvariableop_37_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpWassignvariableop_38_adam_bidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpZassignvariableop_39_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpdassignvariableop_40_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpXassignvariableop_41_adam_bidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_394312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_394312___redundant_placeholder04
0while_while_cond_394312___redundant_placeholder14
0while_while_cond_394312___redundant_placeholder24
0while_while_cond_394312___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
'backward_simple_rnn_1_while_cond_392891H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2J
Fbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_392891___redundant_placeholder0`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_392891___redundant_placeholder1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_392891___redundant_placeholder2`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_392891___redundant_placeholder3(
$backward_simple_rnn_1_while_identity
?
 backward_simple_rnn_1/while/LessLess'backward_simple_rnn_1_while_placeholderFbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_1/while/IdentityIdentity$backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?i
?
6bidirectional_1_forward_simple_rnn_1_while_body_392142f
bbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_loop_counterl
hbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations:
6bidirectional_1_forward_simple_rnn_1_while_placeholder<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_1<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_2<
8bidirectional_1_forward_simple_rnn_1_while_placeholder_3e
abidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1_0?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0p
^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@m
_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@r
`bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@7
3bidirectional_1_forward_simple_rnn_1_while_identity9
5bidirectional_1_forward_simple_rnn_1_while_identity_19
5bidirectional_1_forward_simple_rnn_1_while_identity_29
5bidirectional_1_forward_simple_rnn_1_while_identity_39
5bidirectional_1_forward_simple_rnn_1_while_identity_49
5bidirectional_1_forward_simple_rnn_1_while_identity_5c
_bidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensorn
\bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@k
]bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@p
^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Tbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Sbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Ubidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
\bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Nbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_06bidirectional_1_forward_simple_rnn_1_while_placeholderebidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
^bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Pbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_06bidirectional_1_forward_simple_rnn_1_while_placeholdergbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Sbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Dbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulUbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0[bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Tbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
Ebidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAddNbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0\bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ubidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp`bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Fbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul8bidirectional_1_forward_simple_rnn_1_while_placeholder_3]bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Abidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2Nbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0Pbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Bbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanhEbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
9bidirectional_1/forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/bidirectional_1/forward_simple_rnn_1/while/TileTileWbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Bbidirectional_1/forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
3bidirectional_1/forward_simple_rnn_1/while/SelectV2SelectV28bidirectional_1/forward_simple_rnn_1/while/Tile:output:0Fbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:08bidirectional_1_forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@?
;bidirectional_1/forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
1bidirectional_1/forward_simple_rnn_1/while/Tile_1TileWbidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Dbidirectional_1/forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
5bidirectional_1/forward_simple_rnn_1/while/SelectV2_1SelectV2:bidirectional_1/forward_simple_rnn_1/while/Tile_1:output:0Fbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:08bidirectional_1_forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
Obidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem8bidirectional_1_forward_simple_rnn_1_while_placeholder_16bidirectional_1_forward_simple_rnn_1_while_placeholder<bidirectional_1/forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???r
0bidirectional_1/forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
.bidirectional_1/forward_simple_rnn_1/while/addAddV26bidirectional_1_forward_simple_rnn_1_while_placeholder9bidirectional_1/forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: t
2bidirectional_1/forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
0bidirectional_1/forward_simple_rnn_1/while/add_1AddV2bbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_loop_counter;bidirectional_1/forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
3bidirectional_1/forward_simple_rnn_1/while/IdentityIdentity4bidirectional_1/forward_simple_rnn_1/while/add_1:z:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_1Identityhbidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations0^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_2Identity2bidirectional_1/forward_simple_rnn_1/while/add:z:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_3Identity_bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
5bidirectional_1/forward_simple_rnn_1/while/Identity_4Identity<bidirectional_1/forward_simple_rnn_1/while/SelectV2:output:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
5bidirectional_1/forward_simple_rnn_1/while/Identity_5Identity>bidirectional_1/forward_simple_rnn_1/while/SelectV2_1:output:00^bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
/bidirectional_1/forward_simple_rnn_1/while/NoOpNoOpU^bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpT^bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpV^bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
_bidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1abidirectional_1_forward_simple_rnn_1_while_bidirectional_1_forward_simple_rnn_1_strided_slice_1_0"s
3bidirectional_1_forward_simple_rnn_1_while_identity<bidirectional_1/forward_simple_rnn_1/while/Identity:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_1>bidirectional_1/forward_simple_rnn_1/while/Identity_1:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_2>bidirectional_1/forward_simple_rnn_1/while/Identity_2:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_3>bidirectional_1/forward_simple_rnn_1/while/Identity_3:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_4>bidirectional_1/forward_simple_rnn_1/while/Identity_4:output:0"w
5bidirectional_1_forward_simple_rnn_1_while_identity_5>bidirectional_1/forward_simple_rnn_1/while/Identity_5:output:0"?
]bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource`bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
\bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource^bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Tbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpTbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Sbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpSbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Ubidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpUbidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_sequential_1_layer_call_fn_391486
embedding_1_input
unknown:	?*@
	unknown_0:@@
	unknown_1:@
	unknown_2:@@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:	?@
	unknown_7:@
	unknown_8:@
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_391461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_1_input
?,
?
while_body_390616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_10/MatMul/ReadVariableOp?0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
&forward_simple_rnn_1_while_cond_391578F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3H
Dforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391578___redundant_placeholder0^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391578___redundant_placeholder1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391578___redundant_placeholder2^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391578___redundant_placeholder3^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391578___redundant_placeholder4'
#forward_simple_rnn_1_while_identity
?
forward_simple_rnn_1/while/LessLess&forward_simple_rnn_1_while_placeholderDforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_1/while/IdentityIdentity#forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?!
?
while_body_390041
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_10_390063_0:@@/
!while_simple_rnn_cell_10_390065_0:@3
!while_simple_rnn_cell_10_390067_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_10_390063:@@-
while_simple_rnn_cell_10_390065:@1
while_simple_rnn_cell_10_390067:@@??0while/simple_rnn_cell_10/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
0while/simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_10_390063_0!while_simple_rnn_cell_10_390065_0!while_simple_rnn_cell_10_390067_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390028?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity9while/simple_rnn_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????@

while/NoOpNoOp1^while/simple_rnn_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_10_390063!while_simple_rnn_cell_10_390063_0"D
while_simple_rnn_cell_10_390065!while_simple_rnn_cell_10_390065_0"D
while_simple_rnn_cell_10_390067!while_simple_rnn_cell_10_390067_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2d
0while/simple_rnn_cell_10/StatefulPartitionedCall0while/simple_rnn_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
6__inference_backward_simple_rnn_1_layer_call_fn_394247
inputs_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390559o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
??
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394489
inputs_0C
1simple_rnn_cell_11_matmul_readvariableop_resource:@@@
2simple_rnn_cell_11_biasadd_readvariableop_resource:@E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_11/BiasAdd/ReadVariableOp?(simple_rnn_cell_11/MatMul/ReadVariableOp?*simple_rnn_cell_11/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_11/TanhTanhsimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_394423*
condR
while_cond_394422*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_394816

inputs
states_00
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?
?
while_cond_390199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390199___redundant_placeholder04
0while_while_cond_390199___redundant_placeholder14
0while_while_cond_390199___redundant_placeholder24
0while_while_cond_390199___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?,
?
while_body_394643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_11/MatMul/ReadVariableOp?0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_11/TanhTanh while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_11/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_390881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390881___redundant_placeholder04
0while_while_cond_390881___redundant_placeholder14
0while_while_cond_390881___redundant_placeholder24
0while_while_cond_390881___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?

?
3__inference_simple_rnn_cell_10_layer_call_fn_394723

inputs
states_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_390028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?
?
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_394754

inputs
states_00
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?

?
3__inference_simple_rnn_cell_11_layer_call_fn_394799

inputs
states_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?
?
while_cond_394422
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_394422___redundant_placeholder04
0while_while_cond_394422___redundant_placeholder14
0while_while_cond_394422___redundant_placeholder24
0while_while_cond_394422___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_394050
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_394050___redundant_placeholder04
0while_while_cond_394050___redundant_placeholder14
0while_while_cond_394050___redundant_placeholder24
0while_while_cond_394050___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
'backward_simple_rnn_1_while_cond_393626H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3J
Fbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393626___redundant_placeholder0`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393626___redundant_placeholder1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393626___redundant_placeholder2`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393626___redundant_placeholder3`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393626___redundant_placeholder4(
$backward_simple_rnn_1_while_identity
?
 backward_simple_rnn_1/while/LessLess'backward_simple_rnn_1_while_placeholderFbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_1/while/IdentityIdentity$backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?

?
3__inference_simple_rnn_cell_11_layer_call_fn_394785

inputs
states_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????@:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0
?@
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394599

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource:@@@
2simple_rnn_cell_11_biasadd_readvariableop_resource:@E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_11/BiasAdd/ReadVariableOp?(simple_rnn_cell_11/MatMul/ReadVariableOp?*simple_rnn_cell_11/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'????????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_11/TanhTanhsimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_394533*
condR
while_cond_394532*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?@
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390948

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource:@@@
2simple_rnn_cell_11_biasadd_readvariableop_resource:@E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity??)simple_rnn_cell_11/BiasAdd/ReadVariableOp?(simple_rnn_cell_11/MatMul/ReadVariableOp?*simple_rnn_cell_11/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'????????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@m
simple_rnn_cell_11/TanhTanhsimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390882*
condR
while_cond_390881*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
'backward_simple_rnn_1_while_cond_393107H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2J
Fbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393107___redundant_placeholder0`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393107___redundant_placeholder1`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393107___redundant_placeholder2`
\backward_simple_rnn_1_while_backward_simple_rnn_1_while_cond_393107___redundant_placeholder3(
$backward_simple_rnn_1_while_identity
?
 backward_simple_rnn_1/while/LessLess'backward_simple_rnn_1_while_placeholderFbackward_simple_rnn_1_while_less_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: w
$backward_simple_rnn_1/while/IdentityIdentity$backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?

7bidirectional_1_backward_simple_rnn_1_while_cond_392273h
dbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_loop_countern
jbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations;
7bidirectional_1_backward_simple_rnn_1_while_placeholder=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_1=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_2=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_3j
fbidirectional_1_backward_simple_rnn_1_while_less_bidirectional_1_backward_simple_rnn_1_strided_slice_1?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392273___redundant_placeholder0?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392273___redundant_placeholder1?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392273___redundant_placeholder2?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392273___redundant_placeholder3?
|bidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_cond_392273___redundant_placeholder48
4bidirectional_1_backward_simple_rnn_1_while_identity
?
0bidirectional_1/backward_simple_rnn_1/while/LessLess7bidirectional_1_backward_simple_rnn_1_while_placeholderfbidirectional_1_backward_simple_rnn_1_while_less_bidirectional_1_backward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: ?
4bidirectional_1/backward_simple_rnn_1/while/IdentityIdentity4bidirectional_1/backward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "u
4bidirectional_1_backward_simple_rnn_1_while_identity=bidirectional_1/backward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
??
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391794

inputs
mask
X
Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@U
Gforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@V
Hbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity???backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?backward_simple_rnn_1/while?>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp??forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/whileP
forward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"forward_simple_rnn_1/strided_sliceStridedSlice#forward_simple_rnn_1/Shape:output:01forward_simple_rnn_1/strided_slice/stack:output:03forward_simple_rnn_1/strided_slice/stack_1:output:03forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
!forward_simple_rnn_1/zeros/packedPack+forward_simple_rnn_1/strided_slice:output:0,forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
forward_simple_rnn_1/zerosFill*forward_simple_rnn_1/zeros/packed:output:0)forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@x
#forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
forward_simple_rnn_1/transpose	Transposeinputs,forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@n
forward_simple_rnn_1/Shape_1Shape"forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_1StridedSlice%forward_simple_rnn_1/Shape_1:output:03forward_simple_rnn_1/strided_slice_1/stack:output:05forward_simple_rnn_1/strided_slice_1/stack_1:output:05forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
forward_simple_rnn_1/ExpandDims
ExpandDimsmask,forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????z
%forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_1	Transpose(forward_simple_rnn_1/ExpandDims:output:0.forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????{
0forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"forward_simple_rnn_1/TensorArrayV2TensorListReserve9forward_simple_rnn_1/TensorArrayV2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Jforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
<forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_1/transpose:y:0Sforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???t
*forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_2StridedSlice"forward_simple_rnn_1/transpose:y:03forward_simple_rnn_1/strided_slice_2/stack:output:05forward_simple_rnn_1/strided_slice_2/stack_1:output:05forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
.forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul-forward_simple_rnn_1/strided_slice_2:output:0Eforward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAdd8forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Fforward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
0forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul#forward_simple_rnn_1/zeros:output:0Gforward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+forward_simple_rnn_1/simple_rnn_cell_10/addAddV28forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0:forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
,forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
$forward_simple_rnn_1/TensorArrayV2_1TensorListReserve;forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???[
forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
$forward_simple_rnn_1/TensorArrayV2_2TensorListReserve;forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Lforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
>forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor$forward_simple_rnn_1/transpose_1:y:0Uforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
forward_simple_rnn_1/zeros_like	ZerosLike0forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@x
-forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????i
'forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
forward_simple_rnn_1/whileWhile0forward_simple_rnn_1/while/loop_counter:output:06forward_simple_rnn_1/while/maximum_iterations:output:0"forward_simple_rnn_1/time:output:0-forward_simple_rnn_1/TensorArrayV2_1:handle:0#forward_simple_rnn_1/zeros_like:y:0#forward_simple_rnn_1/zeros:output:0-forward_simple_rnn_1/strided_slice_1:output:0Lforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Nforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&forward_simple_rnn_1_while_body_391579*2
cond*R(
&forward_simple_rnn_1_while_cond_391578*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Eforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
7forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_1/while:output:3Nforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0}
*forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
,forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_3StridedSlice@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_1/strided_slice_3/stack:output:05forward_simple_rnn_1/strided_slice_3/stack_1:output:05forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskz
%forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_2	Transpose@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@Q
backward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#backward_simple_rnn_1/strided_sliceStridedSlice$backward_simple_rnn_1/Shape:output:02backward_simple_rnn_1/strided_slice/stack:output:04backward_simple_rnn_1/strided_slice/stack_1:output:04backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"backward_simple_rnn_1/zeros/packedPack,backward_simple_rnn_1/strided_slice:output:0-backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
backward_simple_rnn_1/zerosFill+backward_simple_rnn_1/zeros/packed:output:0*backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@y
$backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
backward_simple_rnn_1/transpose	Transposeinputs-backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@p
backward_simple_rnn_1/Shape_1Shape#backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_1StridedSlice&backward_simple_rnn_1/Shape_1:output:04backward_simple_rnn_1/strided_slice_1/stack:output:06backward_simple_rnn_1/strided_slice_1/stack_1:output:06backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 backward_simple_rnn_1/ExpandDims
ExpandDimsmask-backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????{
&backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_1	Transpose)backward_simple_rnn_1/ExpandDims:output:0/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????|
1backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#backward_simple_rnn_1/TensorArrayV2TensorListReserve:backward_simple_rnn_1/TensorArrayV2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???n
$backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
backward_simple_rnn_1/ReverseV2	ReverseV2#backward_simple_rnn_1/transpose:y:0-backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
Kbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
=backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_1/ReverseV2:output:0Tbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_2StridedSlice#backward_simple_rnn_1/transpose:y:04backward_simple_rnn_1/strided_slice_2/stack:output:06backward_simple_rnn_1/strided_slice_2/stack_1:output:06backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul.backward_simple_rnn_1/strided_slice_2:output:0Fbackward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAdd9backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Gbackward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
1backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul$backward_simple_rnn_1/zeros:output:0Hbackward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,backward_simple_rnn_1/simple_rnn_cell_11/addAddV29backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0;backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
-backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh0backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%backward_simple_rnn_1/TensorArrayV2_1TensorListReserve<backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
&backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
!backward_simple_rnn_1/ReverseV2_1	ReverseV2%backward_simple_rnn_1/transpose_1:y:0/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :??????????????????~
3backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%backward_simple_rnn_1/TensorArrayV2_2TensorListReserve<backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Mbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
?backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor*backward_simple_rnn_1/ReverseV2_1:output:0Vbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
 backward_simple_rnn_1/zeros_like	ZerosLike1backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@y
.backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
backward_simple_rnn_1/whileWhile1backward_simple_rnn_1/while/loop_counter:output:07backward_simple_rnn_1/while/maximum_iterations:output:0#backward_simple_rnn_1/time:output:0.backward_simple_rnn_1/TensorArrayV2_1:handle:0$backward_simple_rnn_1/zeros_like:y:0$backward_simple_rnn_1/zeros:output:0.backward_simple_rnn_1/strided_slice_1:output:0Mbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Obackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'backward_simple_rnn_1_while_body_391711*3
cond+R)
'backward_simple_rnn_1_while_cond_391710*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Fbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_1/while:output:3Obackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0~
+backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_3StridedSliceAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_1/strided_slice_3/stack:output:06backward_simple_rnn_1/strided_slice_3/stack_1:output:06backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_2	TransposeAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2-forward_simple_rnn_1/strided_slice_3:output:0.backward_simple_rnn_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp@^backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?^backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpA^backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp^backward_simple_rnn_1/while?^forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>^forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp@^forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp^forward_simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????????????@:??????????????????: : : : : : 2?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2:
backward_simple_rnn_1/whilebackward_simple_rnn_1/while2?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp28
forward_simple_rnn_1/whileforward_simple_rnn_1/while:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?	
?
0__inference_bidirectional_1_layer_call_fn_392708
inputs_0
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391109p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?
?
,__inference_embedding_1_layer_call_fn_392664

inputs
unknown:	?*@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_391141|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
&forward_simple_rnn_1_while_cond_391197F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3H
Dforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391197___redundant_placeholder0^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391197___redundant_placeholder1^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391197___redundant_placeholder2^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391197___redundant_placeholder3^
Zforward_simple_rnn_1_while_forward_simple_rnn_1_while_cond_391197___redundant_placeholder4'
#forward_simple_rnn_1_while_identity
?
forward_simple_rnn_1/while/LessLess&forward_simple_rnn_1_while_placeholderDforward_simple_rnn_1_while_less_forward_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: u
#forward_simple_rnn_1/while/IdentityIdentity#forward_simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :?????????@:?????????@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
??
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393176
inputs_0X
Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@U
Gforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@V
Hbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity???backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?backward_simple_rnn_1/while?>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp??forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/whileR
forward_simple_rnn_1/ShapeShapeinputs_0*
T0*
_output_shapes
:r
(forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"forward_simple_rnn_1/strided_sliceStridedSlice#forward_simple_rnn_1/Shape:output:01forward_simple_rnn_1/strided_slice/stack:output:03forward_simple_rnn_1/strided_slice/stack_1:output:03forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
!forward_simple_rnn_1/zeros/packedPack+forward_simple_rnn_1/strided_slice:output:0,forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
forward_simple_rnn_1/zerosFill*forward_simple_rnn_1/zeros/packed:output:0)forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@x
#forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
forward_simple_rnn_1/transpose	Transposeinputs_0,forward_simple_rnn_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????n
forward_simple_rnn_1/Shape_1Shape"forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_1StridedSlice%forward_simple_rnn_1/Shape_1:output:03forward_simple_rnn_1/strided_slice_1/stack:output:05forward_simple_rnn_1/strided_slice_1/stack_1:output:05forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"forward_simple_rnn_1/TensorArrayV2TensorListReserve9forward_simple_rnn_1/TensorArrayV2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Jforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
<forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_1/transpose:y:0Sforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???t
*forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_2StridedSlice"forward_simple_rnn_1/transpose:y:03forward_simple_rnn_1/strided_slice_2/stack:output:05forward_simple_rnn_1/strided_slice_2/stack_1:output:05forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
.forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul-forward_simple_rnn_1/strided_slice_2:output:0Eforward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAdd8forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Fforward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
0forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul#forward_simple_rnn_1/zeros:output:0Gforward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+forward_simple_rnn_1/simple_rnn_cell_10/addAddV28forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0:forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
,forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
$forward_simple_rnn_1/TensorArrayV2_1TensorListReserve;forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???[
forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????i
'forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
forward_simple_rnn_1/whileWhile0forward_simple_rnn_1/while/loop_counter:output:06forward_simple_rnn_1/while/maximum_iterations:output:0"forward_simple_rnn_1/time:output:0-forward_simple_rnn_1/TensorArrayV2_1:handle:0#forward_simple_rnn_1/zeros:output:0-forward_simple_rnn_1/strided_slice_1:output:0Lforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&forward_simple_rnn_1_while_body_393002*2
cond*R(
&forward_simple_rnn_1_while_cond_393001*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
Eforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
7forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_1/while:output:3Nforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0}
*forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
,forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_3StridedSlice@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_1/strided_slice_3/stack:output:05forward_simple_rnn_1/strided_slice_3/stack_1:output:05forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskz
%forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_1	Transpose@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@S
backward_simple_rnn_1/ShapeShapeinputs_0*
T0*
_output_shapes
:s
)backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#backward_simple_rnn_1/strided_sliceStridedSlice$backward_simple_rnn_1/Shape:output:02backward_simple_rnn_1/strided_slice/stack:output:04backward_simple_rnn_1/strided_slice/stack_1:output:04backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"backward_simple_rnn_1/zeros/packedPack,backward_simple_rnn_1/strided_slice:output:0-backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
backward_simple_rnn_1/zerosFill+backward_simple_rnn_1/zeros/packed:output:0*backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@y
$backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
backward_simple_rnn_1/transpose	Transposeinputs_0-backward_simple_rnn_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????p
backward_simple_rnn_1/Shape_1Shape#backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_1StridedSlice&backward_simple_rnn_1/Shape_1:output:04backward_simple_rnn_1/strided_slice_1/stack:output:06backward_simple_rnn_1/strided_slice_1/stack_1:output:06backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#backward_simple_rnn_1/TensorArrayV2TensorListReserve:backward_simple_rnn_1/TensorArrayV2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???n
$backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
backward_simple_rnn_1/ReverseV2	ReverseV2#backward_simple_rnn_1/transpose:y:0-backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Kbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
=backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_1/ReverseV2:output:0Tbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_2StridedSlice#backward_simple_rnn_1/transpose:y:04backward_simple_rnn_1/strided_slice_2/stack:output:06backward_simple_rnn_1/strided_slice_2/stack_1:output:06backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul.backward_simple_rnn_1/strided_slice_2:output:0Fbackward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAdd9backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Gbackward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
1backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul$backward_simple_rnn_1/zeros:output:0Hbackward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,backward_simple_rnn_1/simple_rnn_cell_11/addAddV29backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0;backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
-backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh0backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%backward_simple_rnn_1/TensorArrayV2_1TensorListReserve<backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
backward_simple_rnn_1/whileWhile1backward_simple_rnn_1/while/loop_counter:output:07backward_simple_rnn_1/while/maximum_iterations:output:0#backward_simple_rnn_1/time:output:0.backward_simple_rnn_1/TensorArrayV2_1:handle:0$backward_simple_rnn_1/zeros:output:0.backward_simple_rnn_1/strided_slice_1:output:0Mbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'backward_simple_rnn_1_while_body_393108*3
cond+R)
'backward_simple_rnn_1_while_cond_393107*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
Fbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_1/while:output:3Obackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0~
+backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_3StridedSliceAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_1/strided_slice_3/stack:output:06backward_simple_rnn_1/strided_slice_3/stack_1:output:06backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_1	TransposeAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2-forward_simple_rnn_1/strided_slice_3:output:0.backward_simple_rnn_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp@^backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?^backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpA^backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp^backward_simple_rnn_1/while?^forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>^forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp@^forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp^forward_simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2:
backward_simple_rnn_1/whilebackward_simple_rnn_1/while2?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp28
forward_simple_rnn_1/whileforward_simple_rnn_1/while:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?	
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_391141

inputs*
embedding_lookup_391135:	?*@
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_391135Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/391135*4
_output_shapes"
 :??????????????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/391135*4
_output_shapes"
 :??????????????????@?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
while_cond_393834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_393834___redundant_placeholder04
0while_while_cond_393834___redundant_placeholder14
0while_while_cond_393834___redundant_placeholder24
0while_while_cond_393834___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?U
?
&forward_simple_rnn_1_while_body_393495F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2,
(forward_simple_rnn_1_while_placeholder_3E
Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0?
}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@]
Oforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_1_while_identity)
%forward_simple_rnn_1_while_identity_1)
%forward_simple_rnn_1_while_identity_2)
%forward_simple_rnn_1_while_identity_3)
%forward_simple_rnn_1_while_identity_4)
%forward_simple_rnn_1_while_identity_5C
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor^
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@[
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
Lforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
>forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderUforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Nforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderWforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
4forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulEforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
5forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd>forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0Lforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
6forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul(forward_simple_rnn_1_while_placeholder_3Mforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2>forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0@forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanh5forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@z
)forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
forward_simple_rnn_1/while/TileTileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:02forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
#forward_simple_rnn_1/while/SelectV2SelectV2(forward_simple_rnn_1/while/Tile:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@|
+forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
!forward_simple_rnn_1/while/Tile_1TileGforward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:04forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
%forward_simple_rnn_1/while/SelectV2_1SelectV2*forward_simple_rnn_1/while/Tile_1:output:06forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0(forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_1_while_placeholder_1&forward_simple_rnn_1_while_placeholder,forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???b
 forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
forward_simple_rnn_1/while/addAddV2&forward_simple_rnn_1_while_placeholder)forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
 forward_simple_rnn_1/while/add_1AddV2Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counter+forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
#forward_simple_rnn_1/while/IdentityIdentity$forward_simple_rnn_1/while/add_1:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_1IdentityHforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_2Identity"forward_simple_rnn_1/while/add:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_3IdentityOforward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_4Identity,forward_simple_rnn_1/while/SelectV2:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
%forward_simple_rnn_1/while/Identity_5Identity.forward_simple_rnn_1/while/SelectV2_1:output:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
forward_simple_rnn_1/while/NoOpNoOpE^forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpF^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0"S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0"W
%forward_simple_rnn_1_while_identity_1.forward_simple_rnn_1/while/Identity_1:output:0"W
%forward_simple_rnn_1_while_identity_2.forward_simple_rnn_1/while/Identity_2:output:0"W
%forward_simple_rnn_1_while_identity_3.forward_simple_rnn_1/while/Identity_3:output:0"W
%forward_simple_rnn_1_while_identity_4.forward_simple_rnn_1/while/Identity_4:output:0"W
%forward_simple_rnn_1_while_identity_5.forward_simple_rnn_1/while/Identity_5:output:0"?
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourceOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcePforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpDforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpCforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpEforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
5__inference_forward_simple_rnn_1_layer_call_fn_393760
inputs_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390104o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?@
?
&forward_simple_rnn_1_while_body_393002F
Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counterL
Hforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations*
&forward_simple_rnn_1_while_placeholder,
(forward_simple_rnn_1_while_placeholder_1,
(forward_simple_rnn_1_while_placeholder_2E
Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0?
}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@]
Oforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@b
Pforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@'
#forward_simple_rnn_1_while_identity)
%forward_simple_rnn_1_while_identity_1)
%forward_simple_rnn_1_while_identity_2)
%forward_simple_rnn_1_while_identity_3)
%forward_simple_rnn_1_while_identity_4C
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor^
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@[
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
Lforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
>forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0&forward_simple_rnn_1_while_placeholderUforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
4forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulEforward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Kforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
5forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd>forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0Lforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpPforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
6forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMul(forward_simple_rnn_1_while_placeholder_2Mforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2>forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0@forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanh5forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(forward_simple_rnn_1_while_placeholder_1&forward_simple_rnn_1_while_placeholder6forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:???b
 forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
forward_simple_rnn_1/while/addAddV2&forward_simple_rnn_1_while_placeholder)forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: d
"forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
 forward_simple_rnn_1/while/add_1AddV2Bforward_simple_rnn_1_while_forward_simple_rnn_1_while_loop_counter+forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
#forward_simple_rnn_1/while/IdentityIdentity$forward_simple_rnn_1/while/add_1:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_1IdentityHforward_simple_rnn_1_while_forward_simple_rnn_1_while_maximum_iterations ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_2Identity"forward_simple_rnn_1/while/add:z:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_3IdentityOforward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
%forward_simple_rnn_1/while/Identity_4Identity6forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0 ^forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
forward_simple_rnn_1/while/NoOpNoOpE^forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpF^forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
?forward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1Aforward_simple_rnn_1_while_forward_simple_rnn_1_strided_slice_1_0"S
#forward_simple_rnn_1_while_identity,forward_simple_rnn_1/while/Identity:output:0"W
%forward_simple_rnn_1_while_identity_1.forward_simple_rnn_1/while/Identity_1:output:0"W
%forward_simple_rnn_1_while_identity_2.forward_simple_rnn_1/while/Identity_2:output:0"W
%forward_simple_rnn_1_while_identity_3.forward_simple_rnn_1/while/Identity_3:output:0"W
%forward_simple_rnn_1_while_identity_4.forward_simple_rnn_1/while/Identity_4:output:0"?
Mforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourceOforward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
Nforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcePforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
Lforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceNforward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
{forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor}forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2?
Dforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpDforward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Cforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpCforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Eforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpEforward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?!
?
while_body_390496
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_11_390518_0:@@/
!while_simple_rnn_cell_11_390520_0:@3
!while_simple_rnn_cell_11_390522_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_11_390518:@@-
while_simple_rnn_cell_11_390520:@1
while_simple_rnn_cell_11_390522:@@??0while/simple_rnn_cell_11/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
0while/simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_11_390518_0!while_simple_rnn_cell_11_390520_0!while_simple_rnn_cell_11_390522_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390442?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_11/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity9while/simple_rnn_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????@

while/NoOpNoOp1^while/simple_rnn_cell_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_11_390518!while_simple_rnn_cell_11_390518_0"D
while_simple_rnn_cell_11_390520!while_simple_rnn_cell_11_390520_0"D
while_simple_rnn_cell_11_390522!while_simple_rnn_cell_11_390522_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2d
0while/simple_rnn_cell_11/StatefulPartitionedCall0while/simple_rnn_cell_11/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_390615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_390615___redundant_placeholder04
0while_while_cond_390615___redundant_placeholder14
0while_while_cond_390615___redundant_placeholder24
0while_while_cond_390615___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
??
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393710

inputs
mask
X
Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@U
Gforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@Z
Hforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@Y
Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@V
Hbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@[
Ibackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@
identity???backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?backward_simple_rnn_1/while?>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp??forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/whileP
forward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:r
(forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"forward_simple_rnn_1/strided_sliceStridedSlice#forward_simple_rnn_1/Shape:output:01forward_simple_rnn_1/strided_slice/stack:output:03forward_simple_rnn_1/strided_slice/stack_1:output:03forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
!forward_simple_rnn_1/zeros/packedPack+forward_simple_rnn_1/strided_slice:output:0,forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
forward_simple_rnn_1/zerosFill*forward_simple_rnn_1/zeros/packed:output:0)forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@x
#forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
forward_simple_rnn_1/transpose	Transposeinputs,forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@n
forward_simple_rnn_1/Shape_1Shape"forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:t
*forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_1StridedSlice%forward_simple_rnn_1/Shape_1:output:03forward_simple_rnn_1/strided_slice_1/stack:output:05forward_simple_rnn_1/strided_slice_1/stack_1:output:05forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
forward_simple_rnn_1/ExpandDims
ExpandDimsmask,forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????z
%forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_1	Transpose(forward_simple_rnn_1/ExpandDims:output:0.forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????{
0forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"forward_simple_rnn_1/TensorArrayV2TensorListReserve9forward_simple_rnn_1/TensorArrayV2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Jforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
<forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"forward_simple_rnn_1/transpose:y:0Sforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???t
*forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_2StridedSlice"forward_simple_rnn_1/transpose:y:03forward_simple_rnn_1/strided_slice_2/stack:output:05forward_simple_rnn_1/strided_slice_2/stack_1:output:05forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
.forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul-forward_simple_rnn_1/strided_slice_2:output:0Eforward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAdd8forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Fforward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
0forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul#forward_simple_rnn_1/zeros:output:0Gforward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+forward_simple_rnn_1/simple_rnn_cell_10/addAddV28forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0:forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
,forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
2forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
$forward_simple_rnn_1/TensorArrayV2_1TensorListReserve;forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???[
forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
$forward_simple_rnn_1/TensorArrayV2_2TensorListReserve;forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0-forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Lforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
>forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor$forward_simple_rnn_1/transpose_1:y:0Uforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
forward_simple_rnn_1/zeros_like	ZerosLike0forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@x
-forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????i
'forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
forward_simple_rnn_1/whileWhile0forward_simple_rnn_1/while/loop_counter:output:06forward_simple_rnn_1/while/maximum_iterations:output:0"forward_simple_rnn_1/time:output:0-forward_simple_rnn_1/TensorArrayV2_1:handle:0#forward_simple_rnn_1/zeros_like:y:0#forward_simple_rnn_1/zeros:output:0-forward_simple_rnn_1/strided_slice_1:output:0Lforward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Nforward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Fforward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceGforward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceHforward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&forward_simple_rnn_1_while_body_393495*2
cond*R(
&forward_simple_rnn_1_while_cond_393494*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Eforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
7forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack#forward_simple_rnn_1/while:output:3Nforward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0}
*forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
,forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$forward_simple_rnn_1/strided_slice_3StridedSlice@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03forward_simple_rnn_1/strided_slice_3/stack:output:05forward_simple_rnn_1/strided_slice_3/stack_1:output:05forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskz
%forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
 forward_simple_rnn_1/transpose_2	Transpose@forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@Q
backward_simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:s
)backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#backward_simple_rnn_1/strided_sliceStridedSlice$backward_simple_rnn_1/Shape:output:02backward_simple_rnn_1/strided_slice/stack:output:04backward_simple_rnn_1/strided_slice/stack_1:output:04backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"backward_simple_rnn_1/zeros/packedPack,backward_simple_rnn_1/strided_slice:output:0-backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
backward_simple_rnn_1/zerosFill+backward_simple_rnn_1/zeros/packed:output:0*backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@y
$backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
backward_simple_rnn_1/transpose	Transposeinputs-backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@p
backward_simple_rnn_1/Shape_1Shape#backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:u
+backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_1StridedSlice&backward_simple_rnn_1/Shape_1:output:04backward_simple_rnn_1/strided_slice_1/stack:output:06backward_simple_rnn_1/strided_slice_1/stack_1:output:06backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 backward_simple_rnn_1/ExpandDims
ExpandDimsmask-backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????{
&backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_1	Transpose)backward_simple_rnn_1/ExpandDims:output:0/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????|
1backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#backward_simple_rnn_1/TensorArrayV2TensorListReserve:backward_simple_rnn_1/TensorArrayV2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???n
$backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
backward_simple_rnn_1/ReverseV2	ReverseV2#backward_simple_rnn_1/transpose:y:0-backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
Kbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
=backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(backward_simple_rnn_1/ReverseV2:output:0Tbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_2StridedSlice#backward_simple_rnn_1/transpose:y:04backward_simple_rnn_1/strided_slice_2/stack:output:06backward_simple_rnn_1/strided_slice_2/stack_1:output:06backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul.backward_simple_rnn_1/strided_slice_2:output:0Fbackward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAdd9backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Gbackward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
1backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul$backward_simple_rnn_1/zeros:output:0Hbackward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,backward_simple_rnn_1/simple_rnn_cell_11/addAddV29backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0;backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
-backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh0backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%backward_simple_rnn_1/TensorArrayV2_1TensorListReserve<backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
&backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
!backward_simple_rnn_1/ReverseV2_1	ReverseV2%backward_simple_rnn_1/transpose_1:y:0/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :??????????????????~
3backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%backward_simple_rnn_1/TensorArrayV2_2TensorListReserve<backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0.backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Mbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
?backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor*backward_simple_rnn_1/ReverseV2_1:output:0Vbackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
 backward_simple_rnn_1/zeros_like	ZerosLike1backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@y
.backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
backward_simple_rnn_1/whileWhile1backward_simple_rnn_1/while/loop_counter:output:07backward_simple_rnn_1/while/maximum_iterations:output:0#backward_simple_rnn_1/time:output:0.backward_simple_rnn_1/TensorArrayV2_1:handle:0$backward_simple_rnn_1/zeros_like:y:0$backward_simple_rnn_1/zeros:output:0.backward_simple_rnn_1/strided_slice_1:output:0Mbackward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Obackward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Gbackward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceHbackward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceIbackward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'backward_simple_rnn_1_while_body_393627*3
cond+R)
'backward_simple_rnn_1_while_cond_393626*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Fbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack$backward_simple_rnn_1/while:output:3Obackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0~
+backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%backward_simple_rnn_1/strided_slice_3StridedSliceAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:04backward_simple_rnn_1/strided_slice_3/stack:output:06backward_simple_rnn_1/strided_slice_3/stack_1:output:06backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!backward_simple_rnn_1/transpose_2	TransposeAbackward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2-forward_simple_rnn_1/strided_slice_3:output:0.backward_simple_rnn_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp@^backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?^backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpA^backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp^backward_simple_rnn_1/while?^forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>^forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp@^forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp^forward_simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????????????@:??????????????????: : : : : : 2?
?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp>backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp@backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2:
backward_simple_rnn_1/whilebackward_simple_rnn_1/while2?
>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp>forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2~
=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp=forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp28
forward_simple_rnn_1/whileforward_simple_rnn_1/while:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
6__inference_backward_simple_rnn_1_layer_call_fn_394236
inputs_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390398o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?

?
-__inference_sequential_1_layer_call_fn_392083

inputs
unknown:	?*@
	unknown_0:@@
	unknown_1:@
	unknown_2:@@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:	?@
	unknown_7:@
	unknown_8:@
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_391878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?j
?
7bidirectional_1_backward_simple_rnn_1_while_body_392274h
dbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_loop_countern
jbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations;
7bidirectional_1_backward_simple_rnn_1_while_placeholder=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_1=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_2=
9bidirectional_1_backward_simple_rnn_1_while_placeholder_3g
cbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1_0?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0q
_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@n
`bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@s
abidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@8
4bidirectional_1_backward_simple_rnn_1_while_identity:
6bidirectional_1_backward_simple_rnn_1_while_identity_1:
6bidirectional_1_backward_simple_rnn_1_while_identity_2:
6bidirectional_1_backward_simple_rnn_1_while_identity_3:
6bidirectional_1_backward_simple_rnn_1_while_identity_4:
6bidirectional_1_backward_simple_rnn_1_while_identity_5e
abidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensoro
]bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@l
^bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@q
_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ubidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Tbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Vbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
]bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Obidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_07bidirectional_1_backward_simple_rnn_1_while_placeholderfbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
_bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Qbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_07bidirectional_1_backward_simple_rnn_1_while_placeholderhbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Tbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Ebidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulVbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0\bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ubidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp`bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
Fbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAddObidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0]bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Vbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpabidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Gbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul9bidirectional_1_backward_simple_rnn_1_while_placeholder_3^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Bbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2Obidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Qbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Cbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanhFbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
:bidirectional_1/backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
0bidirectional_1/backward_simple_rnn_1/while/TileTileXbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Cbidirectional_1/backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
4bidirectional_1/backward_simple_rnn_1/while/SelectV2SelectV29bidirectional_1/backward_simple_rnn_1/while/Tile:output:0Gbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:09bidirectional_1_backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@?
<bidirectional_1/backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
2bidirectional_1/backward_simple_rnn_1/while/Tile_1TileXbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Ebidirectional_1/backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
6bidirectional_1/backward_simple_rnn_1/while/SelectV2_1SelectV2;bidirectional_1/backward_simple_rnn_1/while/Tile_1:output:0Gbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:09bidirectional_1_backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
Pbidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9bidirectional_1_backward_simple_rnn_1_while_placeholder_17bidirectional_1_backward_simple_rnn_1_while_placeholder=bidirectional_1/backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???s
1bidirectional_1/backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
/bidirectional_1/backward_simple_rnn_1/while/addAddV27bidirectional_1_backward_simple_rnn_1_while_placeholder:bidirectional_1/backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: u
3bidirectional_1/backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
1bidirectional_1/backward_simple_rnn_1/while/add_1AddV2dbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_loop_counter<bidirectional_1/backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
4bidirectional_1/backward_simple_rnn_1/while/IdentityIdentity5bidirectional_1/backward_simple_rnn_1/while/add_1:z:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_1Identityjbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_while_maximum_iterations1^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_2Identity3bidirectional_1/backward_simple_rnn_1/while/add:z:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_3Identity`bidirectional_1/backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
6bidirectional_1/backward_simple_rnn_1/while/Identity_4Identity=bidirectional_1/backward_simple_rnn_1/while/SelectV2:output:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
6bidirectional_1/backward_simple_rnn_1/while/Identity_5Identity?bidirectional_1/backward_simple_rnn_1/while/SelectV2_1:output:01^bidirectional_1/backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
0bidirectional_1/backward_simple_rnn_1/while/NoOpNoOpV^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpU^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpW^bidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
abidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1cbidirectional_1_backward_simple_rnn_1_while_bidirectional_1_backward_simple_rnn_1_strided_slice_1_0"u
4bidirectional_1_backward_simple_rnn_1_while_identity=bidirectional_1/backward_simple_rnn_1/while/Identity:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_1?bidirectional_1/backward_simple_rnn_1/while/Identity_1:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_2?bidirectional_1/backward_simple_rnn_1/while/Identity_2:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_3?bidirectional_1/backward_simple_rnn_1/while/Identity_3:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_4?bidirectional_1/backward_simple_rnn_1/while/Identity_4:output:0"y
6bidirectional_1_backward_simple_rnn_1_while_identity_5?bidirectional_1/backward_simple_rnn_1/while/Identity_5:output:0"?
^bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource`bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceabidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
]bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_bidirectional_1_backward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?bidirectional_1_backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Ubidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpUbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Tbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpTbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Vbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpVbidirectional_1/backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_392370

inputs6
#embedding_1_embedding_lookup_392087:	?*@h
Vbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource:@@e
Wbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource:@j
Xbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@i
Wbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource:@@f
Xbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource:@k
Ybidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@9
&dense_2_matmul_readvariableop_resource:	?@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:
identity??Obidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Nbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp?Pbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp?+bidirectional_1/backward_simple_rnn_1/while?Nbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp?Mbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp?Obidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp?*bidirectional_1/forward_simple_rnn_1/while?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupj
embedding_1/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_392087embedding_1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/392087*4
_output_shapes"
 :??????????????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/392087*4
_output_shapes"
 :??????????????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@[
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:???????????????????
*bidirectional_1/forward_simple_rnn_1/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
8bidirectional_1/forward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:bidirectional_1/forward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:bidirectional_1/forward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2bidirectional_1/forward_simple_rnn_1/strided_sliceStridedSlice3bidirectional_1/forward_simple_rnn_1/Shape:output:0Abidirectional_1/forward_simple_rnn_1/strided_slice/stack:output:0Cbidirectional_1/forward_simple_rnn_1/strided_slice/stack_1:output:0Cbidirectional_1/forward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3bidirectional_1/forward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
1bidirectional_1/forward_simple_rnn_1/zeros/packedPack;bidirectional_1/forward_simple_rnn_1/strided_slice:output:0<bidirectional_1/forward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:u
0bidirectional_1/forward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
*bidirectional_1/forward_simple_rnn_1/zerosFill:bidirectional_1/forward_simple_rnn_1/zeros/packed:output:09bidirectional_1/forward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@?
3bidirectional_1/forward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
.bidirectional_1/forward_simple_rnn_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0<bidirectional_1/forward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
,bidirectional_1/forward_simple_rnn_1/Shape_1Shape2bidirectional_1/forward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:?
:bidirectional_1/forward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<bidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4bidirectional_1/forward_simple_rnn_1/strided_slice_1StridedSlice5bidirectional_1/forward_simple_rnn_1/Shape_1:output:0Cbidirectional_1/forward_simple_rnn_1/strided_slice_1/stack:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_1:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
3bidirectional_1/forward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
/bidirectional_1/forward_simple_rnn_1/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0<bidirectional_1/forward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :???????????????????
5bidirectional_1/forward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
0bidirectional_1/forward_simple_rnn_1/transpose_1	Transpose8bidirectional_1/forward_simple_rnn_1/ExpandDims:output:0>bidirectional_1/forward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :???????????????????
@bidirectional_1/forward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2bidirectional_1/forward_simple_rnn_1/TensorArrayV2TensorListReserveIbidirectional_1/forward_simple_rnn_1/TensorArrayV2/element_shape:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Zbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Lbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor2bidirectional_1/forward_simple_rnn_1/transpose:y:0cbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
:bidirectional_1/forward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<bidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4bidirectional_1/forward_simple_rnn_1/strided_slice_2StridedSlice2bidirectional_1/forward_simple_rnn_1/transpose:y:0Cbidirectional_1/forward_simple_rnn_1/strided_slice_2/stack:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_1:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Mbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpVbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMulMatMul=bidirectional_1/forward_simple_rnn_1/strided_slice_2:output:0Ubidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Nbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpWbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
?bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAddBiasAddHbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul:product:0Vbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Obidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpXbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
@bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1MatMul3bidirectional_1/forward_simple_rnn_1/zeros:output:0Wbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
;bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/addAddV2Hbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd:output:0Jbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/TanhTanh?bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
Bbidirectional_1/forward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
4bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1TensorListReserveKbidirectional_1/forward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???k
)bidirectional_1/forward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : ?
Bbidirectional_1/forward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
4bidirectional_1/forward_simple_rnn_1/TensorArrayV2_2TensorListReserveKbidirectional_1/forward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
\bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Nbidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor4bidirectional_1/forward_simple_rnn_1/transpose_1:y:0ebidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
/bidirectional_1/forward_simple_rnn_1/zeros_like	ZerosLike@bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/Tanh:y:0*
T0*'
_output_shapes
:?????????@?
=bidirectional_1/forward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????y
7bidirectional_1/forward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?

*bidirectional_1/forward_simple_rnn_1/whileWhile@bidirectional_1/forward_simple_rnn_1/while/loop_counter:output:0Fbidirectional_1/forward_simple_rnn_1/while/maximum_iterations:output:02bidirectional_1/forward_simple_rnn_1/time:output:0=bidirectional_1/forward_simple_rnn_1/TensorArrayV2_1:handle:03bidirectional_1/forward_simple_rnn_1/zeros_like:y:03bidirectional_1/forward_simple_rnn_1/zeros:output:0=bidirectional_1/forward_simple_rnn_1/strided_slice_1:output:0\bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0^bidirectional_1/forward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Vbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_readvariableop_resourceWbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_biasadd_readvariableop_resourceXbidirectional_1_forward_simple_rnn_1_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *B
body:R8
6bidirectional_1_forward_simple_rnn_1_while_body_392142*B
cond:R8
6bidirectional_1_forward_simple_rnn_1_while_cond_392141*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Ubidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Gbidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack3bidirectional_1/forward_simple_rnn_1/while:output:3^bidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0?
:bidirectional_1/forward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
<bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<bidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4bidirectional_1/forward_simple_rnn_1/strided_slice_3StridedSlicePbidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Cbidirectional_1/forward_simple_rnn_1/strided_slice_3/stack:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_1:output:0Ebidirectional_1/forward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
5bidirectional_1/forward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
0bidirectional_1/forward_simple_rnn_1/transpose_2	TransposePbidirectional_1/forward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0>bidirectional_1/forward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
+bidirectional_1/backward_simple_rnn_1/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
9bidirectional_1/backward_simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;bidirectional_1/backward_simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;bidirectional_1/backward_simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3bidirectional_1/backward_simple_rnn_1/strided_sliceStridedSlice4bidirectional_1/backward_simple_rnn_1/Shape:output:0Bbidirectional_1/backward_simple_rnn_1/strided_slice/stack:output:0Dbidirectional_1/backward_simple_rnn_1/strided_slice/stack_1:output:0Dbidirectional_1/backward_simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4bidirectional_1/backward_simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
2bidirectional_1/backward_simple_rnn_1/zeros/packedPack<bidirectional_1/backward_simple_rnn_1/strided_slice:output:0=bidirectional_1/backward_simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1bidirectional_1/backward_simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+bidirectional_1/backward_simple_rnn_1/zerosFill;bidirectional_1/backward_simple_rnn_1/zeros/packed:output:0:bidirectional_1/backward_simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@?
4bidirectional_1/backward_simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
/bidirectional_1/backward_simple_rnn_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0=bidirectional_1/backward_simple_rnn_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@?
-bidirectional_1/backward_simple_rnn_1/Shape_1Shape3bidirectional_1/backward_simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:?
;bidirectional_1/backward_simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=bidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5bidirectional_1/backward_simple_rnn_1/strided_slice_1StridedSlice6bidirectional_1/backward_simple_rnn_1/Shape_1:output:0Dbidirectional_1/backward_simple_rnn_1/strided_slice_1/stack:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_1:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
4bidirectional_1/backward_simple_rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0bidirectional_1/backward_simple_rnn_1/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0=bidirectional_1/backward_simple_rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :???????????????????
6bidirectional_1/backward_simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1bidirectional_1/backward_simple_rnn_1/transpose_1	Transpose9bidirectional_1/backward_simple_rnn_1/ExpandDims:output:0?bidirectional_1/backward_simple_rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :???????????????????
Abidirectional_1/backward_simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
3bidirectional_1/backward_simple_rnn_1/TensorArrayV2TensorListReserveJbidirectional_1/backward_simple_rnn_1/TensorArrayV2/element_shape:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???~
4bidirectional_1/backward_simple_rnn_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
/bidirectional_1/backward_simple_rnn_1/ReverseV2	ReverseV23bidirectional_1/backward_simple_rnn_1/transpose:y:0=bidirectional_1/backward_simple_rnn_1/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
[bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Mbidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor8bidirectional_1/backward_simple_rnn_1/ReverseV2:output:0dbidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
;bidirectional_1/backward_simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=bidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5bidirectional_1/backward_simple_rnn_1/strided_slice_2StridedSlice3bidirectional_1/backward_simple_rnn_1/transpose:y:0Dbidirectional_1/backward_simple_rnn_1/strided_slice_2/stack:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_1:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
Nbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpWbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMulMatMul>bidirectional_1/backward_simple_rnn_1/strided_slice_2:output:0Vbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Obidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpXbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
@bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAddBiasAddIbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul:product:0Wbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Pbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpYbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Abidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1MatMul4bidirectional_1/backward_simple_rnn_1/zeros:output:0Xbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
<bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/addAddV2Ibidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd:output:0Kbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/TanhTanh@bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
Cbidirectional_1/backward_simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
5bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1TensorListReserveLbidirectional_1/backward_simple_rnn_1/TensorArrayV2_1/element_shape:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
*bidirectional_1/backward_simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : ?
6bidirectional_1/backward_simple_rnn_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: ?
1bidirectional_1/backward_simple_rnn_1/ReverseV2_1	ReverseV25bidirectional_1/backward_simple_rnn_1/transpose_1:y:0?bidirectional_1/backward_simple_rnn_1/ReverseV2_1/axis:output:0*
T0
*4
_output_shapes"
 :???????????????????
Cbidirectional_1/backward_simple_rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5bidirectional_1/backward_simple_rnn_1/TensorArrayV2_2TensorListReserveLbidirectional_1/backward_simple_rnn_1/TensorArrayV2_2/element_shape:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
]bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Obidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor:bidirectional_1/backward_simple_rnn_1/ReverseV2_1:output:0fbidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
0bidirectional_1/backward_simple_rnn_1/zeros_like	ZerosLikeAbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/Tanh:y:0*
T0*'
_output_shapes
:?????????@?
>bidirectional_1/backward_simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????z
8bidirectional_1/backward_simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
+bidirectional_1/backward_simple_rnn_1/whileWhileAbidirectional_1/backward_simple_rnn_1/while/loop_counter:output:0Gbidirectional_1/backward_simple_rnn_1/while/maximum_iterations:output:03bidirectional_1/backward_simple_rnn_1/time:output:0>bidirectional_1/backward_simple_rnn_1/TensorArrayV2_1:handle:04bidirectional_1/backward_simple_rnn_1/zeros_like:y:04bidirectional_1/backward_simple_rnn_1/zeros:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_1:output:0]bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0_bidirectional_1/backward_simple_rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Wbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_readvariableop_resourceXbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_biasadd_readvariableop_resourceYbidirectional_1_backward_simple_rnn_1_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????@:?????????@: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *C
body;R9
7bidirectional_1_backward_simple_rnn_1_while_body_392274*C
cond;R9
7bidirectional_1_backward_simple_rnn_1_while_cond_392273*M
output_shapes<
:: : : : :?????????@:?????????@: : : : : : *
parallel_iterations ?
Vbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
Hbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack4bidirectional_1/backward_simple_rnn_1/while:output:3_bidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0?
;bidirectional_1/backward_simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
=bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=bidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5bidirectional_1/backward_simple_rnn_1/strided_slice_3StridedSliceQbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0Dbidirectional_1/backward_simple_rnn_1/strided_slice_3/stack:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_1:output:0Fbidirectional_1/backward_simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
6bidirectional_1/backward_simple_rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1bidirectional_1/backward_simple_rnn_1/transpose_2	TransposeQbidirectional_1/backward_simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional_1/backward_simple_rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@]
bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
bidirectional_1/concatConcatV2=bidirectional_1/forward_simple_rnn_1/strided_slice_3:output:0>bidirectional_1/backward_simple_rnn_1/strided_slice_3:output:0$bidirectional_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_2/MatMulMatMulbidirectional_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpP^bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpO^bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpQ^bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp,^bidirectional_1/backward_simple_rnn_1/whileO^bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpN^bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpP^bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp+^bidirectional_1/forward_simple_rnn_1/while^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:??????????????????: : : : : : : : : : : 2?
Obidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOpObidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Nbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOpNbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Pbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOpPbidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/MatMul_1/ReadVariableOp2Z
+bidirectional_1/backward_simple_rnn_1/while+bidirectional_1/backward_simple_rnn_1/while2?
Nbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOpNbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
Mbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOpMbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul/ReadVariableOp2?
Obidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOpObidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/MatMul_1/ReadVariableOp2X
*bidirectional_1/forward_simple_rnn_1/while*bidirectional_1/forward_simple_rnn_1/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_391109

inputs-
forward_simple_rnn_1_391092:@@)
forward_simple_rnn_1_391094:@-
forward_simple_rnn_1_391096:@@.
backward_simple_rnn_1_391099:@@*
backward_simple_rnn_1_391101:@.
backward_simple_rnn_1_391103:@@
identity??-backward_simple_rnn_1/StatefulPartitionedCall?,forward_simple_rnn_1/StatefulPartitionedCall?
,forward_simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_1_391092forward_simple_rnn_1_391094forward_simple_rnn_1_391096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_391078?
-backward_simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_1_391099backward_simple_rnn_1_391101backward_simple_rnn_1_391103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390948M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV25forward_simple_rnn_1/StatefulPartitionedCall:output:06backward_simple_rnn_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp.^backward_simple_rnn_1/StatefulPartitionedCall-^forward_simple_rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2^
-backward_simple_rnn_1/StatefulPartitionedCall-backward_simple_rnn_1/StatefulPartitionedCall2\
,forward_simple_rnn_1/StatefulPartitionedCall,forward_simple_rnn_1/StatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?5
?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_390559

inputs+
simple_rnn_cell_11_390484:@@'
simple_rnn_cell_11_390486:@+
simple_rnn_cell_11_390488:@@
identity??*simple_rnn_cell_11/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????@?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
*simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_11_390484simple_rnn_cell_11_390486simple_rnn_cell_11_390488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_390442n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_11_390484simple_rnn_cell_11_390486simple_rnn_cell_11_390488*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_390496*
condR
while_cond_390495*8
output_shapes'
%: : : : :?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????@{
NoOpNoOp+^simple_rnn_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2X
*simple_rnn_cell_11/StatefulPartitionedCall*simple_rnn_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?,
?
while_body_394313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource:@@F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_11/MatMul/ReadVariableOp?0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@y
while/simple_rnn_cell_11/TanhTanh while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_11/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?V
?
'backward_simple_rnn_1_while_body_391711H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2-
)backward_simple_rnn_1_while_placeholder_3G
Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0?
backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@^
Pbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_1_while_identity*
&backward_simple_rnn_1_while_identity_1*
&backward_simple_rnn_1_while_identity_2*
&backward_simple_rnn_1_while_identity_3*
&backward_simple_rnn_1_while_identity_4*
&backward_simple_rnn_1_while_identity_5E
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@\
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
Mbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
?backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderVbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Obackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Abackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderXbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
5backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulFbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
6backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd?backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0Mbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
7backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul)backward_simple_rnn_1_while_placeholder_3Nbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
2backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2?backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Abackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanh6backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@{
*backward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
 backward_simple_rnn_1/while/TileTileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:03backward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
$backward_simple_rnn_1/while/SelectV2SelectV2)backward_simple_rnn_1/while/Tile:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@}
,backward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
"backward_simple_rnn_1/while/Tile_1TileHbackward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:05backward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
&backward_simple_rnn_1/while/SelectV2_1SelectV2+backward_simple_rnn_1/while/Tile_1:output:07backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0)backward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_1_while_placeholder_1'backward_simple_rnn_1_while_placeholder-backward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???c
!backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
backward_simple_rnn_1/while/addAddV2'backward_simple_rnn_1_while_placeholder*backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!backward_simple_rnn_1/while/add_1AddV2Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counter,backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$backward_simple_rnn_1/while/IdentityIdentity%backward_simple_rnn_1/while/add_1:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_1IdentityJbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_2Identity#backward_simple_rnn_1/while/add:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_3IdentityPbackward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_4Identity-backward_simple_rnn_1/while/SelectV2:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
&backward_simple_rnn_1/while/Identity_5Identity/backward_simple_rnn_1/while/SelectV2_1:output:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
 backward_simple_rnn_1/while/NoOpNoOpF^backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpG^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0"U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0"Y
&backward_simple_rnn_1_while_identity_1/backward_simple_rnn_1/while/Identity_1:output:0"Y
&backward_simple_rnn_1_while_identity_2/backward_simple_rnn_1/while/Identity_2:output:0"Y
&backward_simple_rnn_1_while_identity_3/backward_simple_rnn_1/while/Identity_3:output:0"Y
&backward_simple_rnn_1_while_identity_4/backward_simple_rnn_1/while/Identity_4:output:0"Y
&backward_simple_rnn_1_while_identity_5/backward_simple_rnn_1/while/Identity_5:output:0"?
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcePbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourceObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?backward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpEbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpDbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpFbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_391011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_391011___redundant_placeholder04
0while_while_cond_391011___redundant_placeholder14
0while_while_cond_391011___redundant_placeholder24
0while_while_cond_391011___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
5__inference_forward_simple_rnn_1_layer_call_fn_393771
inputs_0
unknown:@@
	unknown_0:@
	unknown_1:@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_390263o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?y
?
Csequential_1_bidirectional_1_forward_simple_rnn_1_while_body_389752?
|sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_loop_counter?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_maximum_iterationsG
Csequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholderI
Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_1I
Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_2I
Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_3
{sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_strided_slice_1_0?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0}
ksequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@@z
lsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@
msequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@D
@sequential_1_bidirectional_1_forward_simple_rnn_1_while_identityF
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_1F
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_2F
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_3F
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_4F
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_5}
ysequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_strided_slice_1?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor{
isequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource:@@x
jsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@}
ksequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@??asequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp?`sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp?bsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp?
isequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
[sequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0Csequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholderrsequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
ksequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
]sequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0Csequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholdertsequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
`sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpksequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Qsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMulMatMulbsequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0hsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
asequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOplsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
Rsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAddBiasAdd[sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul:product:0isequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
bsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpmsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
Ssequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1MatMulEsequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_3jsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Nsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/addAddV2[sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd:output:0]sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
Osequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/TanhTanhRsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:?????????@?
Fsequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
<sequential_1/bidirectional_1/forward_simple_rnn_1/while/TileTiledsequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Osequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
@sequential_1/bidirectional_1/forward_simple_rnn_1/while/SelectV2SelectV2Esequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile:output:0Ssequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:?????????@?
Hsequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
>sequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile_1Tiledsequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Qsequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:??????????
Bsequential_1/bidirectional_1/forward_simple_rnn_1/while/SelectV2_1SelectV2Gsequential_1/bidirectional_1/forward_simple_rnn_1/while/Tile_1:output:0Ssequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/Tanh:y:0Esequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
\sequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEsequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholder_1Csequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholderIsequential_1/bidirectional_1/forward_simple_rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???
=sequential_1/bidirectional_1/forward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
;sequential_1/bidirectional_1/forward_simple_rnn_1/while/addAddV2Csequential_1_bidirectional_1_forward_simple_rnn_1_while_placeholderFsequential_1/bidirectional_1/forward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: ?
?sequential_1/bidirectional_1/forward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
=sequential_1/bidirectional_1/forward_simple_rnn_1/while/add_1AddV2|sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_loop_counterHsequential_1/bidirectional_1/forward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
@sequential_1/bidirectional_1/forward_simple_rnn_1/while/IdentityIdentityAsequential_1/bidirectional_1/forward_simple_rnn_1/while/add_1:z:0=^sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Bsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_1Identity?sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_while_maximum_iterations=^sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Bsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_2Identity?sequential_1/bidirectional_1/forward_simple_rnn_1/while/add:z:0=^sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Bsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_3Identitylsequential_1/bidirectional_1/forward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
Bsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_4IdentityIsequential_1/bidirectional_1/forward_simple_rnn_1/while/SelectV2:output:0=^sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
Bsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_5IdentityKsequential_1/bidirectional_1/forward_simple_rnn_1/while/SelectV2_1:output:0=^sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
<sequential_1/bidirectional_1/forward_simple_rnn_1/while/NoOpNoOpb^sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpa^sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOpc^sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
@sequential_1_bidirectional_1_forward_simple_rnn_1_while_identityIsequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity:output:0"?
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_1Ksequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_1:output:0"?
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_2Ksequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_2:output:0"?
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_3Ksequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_3:output:0"?
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_4Ksequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_4:output:0"?
Bsequential_1_bidirectional_1_forward_simple_rnn_1_while_identity_5Ksequential_1/bidirectional_1/forward_simple_rnn_1/while/Identity_5:output:0"?
ysequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_strided_slice_1{sequential_1_bidirectional_1_forward_simple_rnn_1_while_sequential_1_bidirectional_1_forward_simple_rnn_1_strided_slice_1_0"?
jsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resourcelsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"?
ksequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resourcemsequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"?
isequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resourceksequential_1_bidirectional_1_forward_simple_rnn_1_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor?sequential_1_bidirectional_1_forward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :?????????@:?????????@: : : : : : 2?
asequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpasequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2?
`sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp`sequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul/ReadVariableOp2?
bsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpbsequential_1/bidirectional_1/forward_simple_rnn_1/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?A
?
'backward_simple_rnn_1_while_body_392892H
Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counterN
Jbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations+
'backward_simple_rnn_1_while_placeholder-
)backward_simple_rnn_1_while_placeholder_1-
)backward_simple_rnn_1_while_placeholder_2G
Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0?
backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0:@@^
Pbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0:@c
Qbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:@@(
$backward_simple_rnn_1_while_identity*
&backward_simple_rnn_1_while_identity_1*
&backward_simple_rnn_1_while_identity_2*
&backward_simple_rnn_1_while_identity_3*
&backward_simple_rnn_1_while_identity_4E
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource:@@\
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource:@a
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:@@??Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp?Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp?Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp?
Mbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????????
?backward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0'backward_simple_rnn_1_while_placeholderVbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
5backward_simple_rnn_1/while/simple_rnn_cell_11/MatMulMatMulFbackward_simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpPbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
6backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAddBiasAdd?backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul:product:0Mbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0?
7backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1MatMul)backward_simple_rnn_1_while_placeholder_2Nbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
2backward_simple_rnn_1/while/simple_rnn_cell_11/addAddV2?backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd:output:0Abackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????@?
3backward_simple_rnn_1/while/simple_rnn_cell_11/TanhTanh6backward_simple_rnn_1/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:?????????@?
@backward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)backward_simple_rnn_1_while_placeholder_1'backward_simple_rnn_1_while_placeholder7backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0*
_output_shapes
: *
element_dtype0:???c
!backward_simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
backward_simple_rnn_1/while/addAddV2'backward_simple_rnn_1_while_placeholder*backward_simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: e
#backward_simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!backward_simple_rnn_1/while/add_1AddV2Dbackward_simple_rnn_1_while_backward_simple_rnn_1_while_loop_counter,backward_simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$backward_simple_rnn_1/while/IdentityIdentity%backward_simple_rnn_1/while/add_1:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_1IdentityJbackward_simple_rnn_1_while_backward_simple_rnn_1_while_maximum_iterations!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_2Identity#backward_simple_rnn_1/while/add:z:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_3IdentityPbackward_simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^backward_simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ?
&backward_simple_rnn_1/while/Identity_4Identity7backward_simple_rnn_1/while/simple_rnn_cell_11/Tanh:y:0!^backward_simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:?????????@?
 backward_simple_rnn_1/while/NoOpNoOpF^backward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpG^backward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Abackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1Cbackward_simple_rnn_1_while_backward_simple_rnn_1_strided_slice_1_0"U
$backward_simple_rnn_1_while_identity-backward_simple_rnn_1/while/Identity:output:0"Y
&backward_simple_rnn_1_while_identity_1/backward_simple_rnn_1/while/Identity_1:output:0"Y
&backward_simple_rnn_1_while_identity_2/backward_simple_rnn_1/while/Identity_2:output:0"Y
&backward_simple_rnn_1_while_identity_3/backward_simple_rnn_1/while/Identity_3:output:0"Y
&backward_simple_rnn_1_while_identity_4/backward_simple_rnn_1/while/Identity_4:output:0"?
Nbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resourcePbackward_simple_rnn_1_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"?
Obackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceQbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"?
Mbackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resourceObackward_simple_rnn_1_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"?
}backward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorbackward_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????@: : : : : 2?
Ebackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpEbackward_simple_rnn_1/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2?
Dbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOpDbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul/ReadVariableOp2?
Fbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpFbackward_simple_rnn_1/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_2_layer_call_fn_393719

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_391438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
embedding_1_inputC
#serving_default_embedding_1_input:0??????????????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratem{m|m}m~m'm?(m?)m?*m?+m?,m?v?v?v?v?v?'v?(v?)v?*v?+v?,v?"
	optimizer
n
0
'1
(2
)3
*4
+5
,6
7
8
9
10"
trackable_list_wrapper
n
0
'1
(2
)3
*4
+5
,6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'	?*@2embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
7cell
8
state_spec
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
=cell
>
state_spec
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_2/kernel
:@2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
 regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
P:N@@2>bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel
Z:X@@2Hbidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel
J:H@2<bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias
Q:O@@2?bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel
[:Y@@2Ibidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel
K:I@2=bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

'kernel
(recurrent_kernel
)bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Xstates
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
9	variables
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

*kernel
+recurrent_kernel
,bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

bstates
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
?	variables
@trainable_variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	htotal
	icount
j	variables
k	keras_api"
_tf_keras_metric
^
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api"
_tf_keras_metric
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
*0
+1
,2"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
^	variables
_trainable_variables
`regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.:,	?*@2Adam/embedding_1/embeddings/m
&:$	?@2Adam/dense_2/kernel/m
:@2Adam/dense_2/bias/m
%:#@2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
U:S@@2EAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/m
_:]@@2OAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/m
O:M@2CAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/m
V:T@@2FAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/m
`:^@@2PAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/m
P:N@2DAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/m
.:,	?*@2Adam/embedding_1/embeddings/v
&:$	?@2Adam/dense_2/kernel/v
:@2Adam/dense_2/bias/v
%:#@2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
U:S@@2EAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/kernel/v
_:]@@2OAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/recurrent_kernel/v
O:M@2CAdam/bidirectional_1/forward_simple_rnn_1/simple_rnn_cell_10/bias/v
V:T@@2FAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/kernel/v
`:^@@2PAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/recurrent_kernel/v
P:N@2DAdam/bidirectional_1/backward_simple_rnn_1/simple_rnn_cell_11/bias/v
?2?
-__inference_sequential_1_layer_call_fn_391486
-__inference_sequential_1_layer_call_fn_392056
-__inference_sequential_1_layer_call_fn_392083
-__inference_sequential_1_layer_call_fn_391930?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_1_layer_call_and_return_conditional_losses_392370
H__inference_sequential_1_layer_call_and_return_conditional_losses_392657
H__inference_sequential_1_layer_call_and_return_conditional_losses_391962
H__inference_sequential_1_layer_call_and_return_conditional_losses_391994?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_389980embedding_1_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_1_layer_call_fn_392664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_1_layer_call_and_return_conditional_losses_392674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_bidirectional_1_layer_call_fn_392691
0__inference_bidirectional_1_layer_call_fn_392708
0__inference_bidirectional_1_layer_call_fn_392726
0__inference_bidirectional_1_layer_call_fn_392744?
???
FullArgSpecO
argsG?D
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults?
p 

 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_392960
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393176
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393443
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393710?
???
FullArgSpecO
argsG?D
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults?
p 

 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_393719?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_393730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_393739?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_393749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_392029embedding_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_forward_simple_rnn_1_layer_call_fn_393760
5__inference_forward_simple_rnn_1_layer_call_fn_393771
5__inference_forward_simple_rnn_1_layer_call_fn_393782
5__inference_forward_simple_rnn_1_layer_call_fn_393793?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_393901
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394009
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394117
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394225?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_backward_simple_rnn_1_layer_call_fn_394236
6__inference_backward_simple_rnn_1_layer_call_fn_394247
6__inference_backward_simple_rnn_1_layer_call_fn_394258
6__inference_backward_simple_rnn_1_layer_call_fn_394269?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394379
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394489
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394599
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394709?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_simple_rnn_cell_10_layer_call_fn_394723
3__inference_simple_rnn_cell_10_layer_call_fn_394737?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_394754
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_394771?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_simple_rnn_cell_11_layer_call_fn_394785
3__inference_simple_rnn_cell_11_layer_call_fn_394799?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_394816
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_394833?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_389980?')(*,+C?@
9?6
4?1
embedding_1_input??????????????????
? "1?.
,
dense_3!?
dense_3??????????
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394379}*,+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "%?"
?
0?????????@
? ?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394489}*,+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "%?"
?
0?????????@
? ?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394599*,+Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "%?"
?
0?????????@
? ?
Q__inference_backward_simple_rnn_1_layer_call_and_return_conditional_losses_394709*,+Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "%?"
?
0?????????@
? ?
6__inference_backward_simple_rnn_1_layer_call_fn_394236p*,+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "??????????@?
6__inference_backward_simple_rnn_1_layer_call_fn_394247p*,+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "??????????@?
6__inference_backward_simple_rnn_1_layer_call_fn_394258r*,+Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "??????????@?
6__inference_backward_simple_rnn_1_layer_call_fn_394269r*,+Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "??????????@?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_392960?')(*,+\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p 

 

 

 
? "&?#
?
0??????????
? ?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393176?')(*,+\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p

 

 

 
? "&?#
?
0??????????
? ?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393443?')(*,+q?n
g?d
-?*
inputs??????????????????@
p 
'?$
mask??????????????????


 

 
? "&?#
?
0??????????
? ?
K__inference_bidirectional_1_layer_call_and_return_conditional_losses_393710?')(*,+q?n
g?d
-?*
inputs??????????????????@
p
'?$
mask??????????????????


 

 
? "&?#
?
0??????????
? ?
0__inference_bidirectional_1_layer_call_fn_392691?')(*,+\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p 

 

 

 
? "????????????
0__inference_bidirectional_1_layer_call_fn_392708?')(*,+\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p

 

 

 
? "????????????
0__inference_bidirectional_1_layer_call_fn_392726?')(*,+q?n
g?d
-?*
inputs??????????????????@
p 
'?$
mask??????????????????


 

 
? "????????????
0__inference_bidirectional_1_layer_call_fn_392744?')(*,+q?n
g?d
-?*
inputs??????????????????@
p
'?$
mask??????????????????


 

 
? "????????????
C__inference_dense_2_layer_call_and_return_conditional_losses_393730]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_2_layer_call_fn_393719P0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_3_layer_call_and_return_conditional_losses_393749\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_393739O/?,
%?"
 ?
inputs?????????@
? "???????????
G__inference_embedding_1_layer_call_and_return_conditional_losses_392674q8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0??????????????????@
? ?
,__inference_embedding_1_layer_call_fn_392664d8?5
.?+
)?&
inputs??????????????????
? "%?"??????????????????@?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_393901}')(O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "%?"
?
0?????????@
? ?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394009}')(O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "%?"
?
0?????????@
? ?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394117')(Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "%?"
?
0?????????@
? ?
P__inference_forward_simple_rnn_1_layer_call_and_return_conditional_losses_394225')(Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "%?"
?
0?????????@
? ?
5__inference_forward_simple_rnn_1_layer_call_fn_393760p')(O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "??????????@?
5__inference_forward_simple_rnn_1_layer_call_fn_393771p')(O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "??????????@?
5__inference_forward_simple_rnn_1_layer_call_fn_393782r')(Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "??????????@?
5__inference_forward_simple_rnn_1_layer_call_fn_393793r')(Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "??????????@?
H__inference_sequential_1_layer_call_and_return_conditional_losses_391962?')(*,+K?H
A?>
4?1
embedding_1_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_391994?')(*,+K?H
A?>
4?1
embedding_1_input??????????????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_392370v')(*,+@?=
6?3
)?&
inputs??????????????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_392657v')(*,+@?=
6?3
)?&
inputs??????????????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_1_layer_call_fn_391486t')(*,+K?H
A?>
4?1
embedding_1_input??????????????????
p 

 
? "???????????
-__inference_sequential_1_layer_call_fn_391930t')(*,+K?H
A?>
4?1
embedding_1_input??????????????????
p

 
? "???????????
-__inference_sequential_1_layer_call_fn_392056i')(*,+@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
-__inference_sequential_1_layer_call_fn_392083i')(*,+@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
$__inference_signature_wrapper_392029?')(*,+X?U
? 
N?K
I
embedding_1_input4?1
embedding_1_input??????????????????"1?.
,
dense_3!?
dense_3??????????
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_394754?')(\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p 
? "R?O
H?E
?
0/0?????????@
$?!
?
0/1/0?????????@
? ?
N__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_394771?')(\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p
? "R?O
H?E
?
0/0?????????@
$?!
?
0/1/0?????????@
? ?
3__inference_simple_rnn_cell_10_layer_call_fn_394723?')(\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p 
? "D?A
?
0?????????@
"?
?
1/0?????????@?
3__inference_simple_rnn_cell_10_layer_call_fn_394737?')(\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p
? "D?A
?
0?????????@
"?
?
1/0?????????@?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_394816?*,+\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p 
? "R?O
H?E
?
0/0?????????@
$?!
?
0/1/0?????????@
? ?
N__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_394833?*,+\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p
? "R?O
H?E
?
0/0?????????@
$?!
?
0/1/0?????????@
? ?
3__inference_simple_rnn_cell_11_layer_call_fn_394785?*,+\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p 
? "D?A
?
0?????????@
"?
?
1/0?????????@?
3__inference_simple_rnn_cell_11_layer_call_fn_394799?*,+\?Y
R?O
 ?
inputs?????????@
'?$
"?
states/0?????????@
p
? "D?A
?
0?????????@
"?
?
1/0?????????@