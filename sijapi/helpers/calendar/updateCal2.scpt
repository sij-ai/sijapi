FasdUAS 1.101.10   ��   ��    k             l     ����  r       	  c     	 
  
 l     ����  n         1    ��
�� 
psxp  l     ����  I    �� ��
�� .earsffdralis        afdr  m     ��
�� afdrcusr��  ��  ��  ��  ��    m    ��
�� 
TEXT 	 o      ���� 0 
homefolder 
homeFolder��  ��        l    ����  r        b        o    ���� 0 
homefolder 
homeFolder  m       �   * w o r k s h o p / s i j a p i / d a t a /  o      ���� 0 exportfolder exportFolder��  ��        l    ����  r        b         o    ���� 0 exportfolder exportFolder   m     ! ! � " "  c a l e n d a r . i c s  o      ���� 0 
exportfile 
exportFile��  ��     # $ # l    %���� % r     & ' & b     ( ) ( o    ���� 0 
homefolder 
homeFolder ) m     * * � + + " c a l e n d a r _ t e m p . i c s ' o      ����  0 tempexportfile tempExportFile��  ��   $  , - , l     ��������  ��  ��   -  . / . l     �� 0 1��   0 ) # Ensure the target directory exists    1 � 2 2 F   E n s u r e   t h e   t a r g e t   d i r e c t o r y   e x i s t s /  3 4 3 l   ' 5���� 5 I   '�� 6��
�� .sysoexecTEXT���     TEXT 6 b    # 7 8 7 m     9 9 � : :  m k d i r   - p   8 n    " ; < ; 1     "��
�� 
strq < o     ���� 0 exportfolder exportFolder��  ��  ��   4  = > = l     ��������  ��  ��   >  ? @ ? l     �� A B��   A < 6 Calculate date range (last 3 months to next 3 months)    B � C C l   C a l c u l a t e   d a t e   r a n g e   ( l a s t   3   m o n t h s   t o   n e x t   3   m o n t h s ) @  D E D l  ( / F���� F r   ( / G H G I  ( -������
�� .misccurdldt    ��� null��  ��   H o      ���� 	0 today  ��  ��   E  I J I l  0 = K���� K r   0 = L M L l  0 9 N���� N \   0 9 O P O o   0 1���� 	0 today   P l  1 8 Q���� Q ]   1 8 R S R m   1 4����  S 1   4 7��
�� 
days��  ��  ��  ��   M o      ���� 0 pastdate pastDate��  ��   J  T U T l  > K V���� V r   > K W X W l  > G Y���� Y [   > G Z [ Z o   > ?���� 	0 today   [ l  ? F \���� \ ]   ? F ] ^ ] m   ? B���� Z ^ 1   B E��
�� 
days��  ��  ��  ��   X o      ���� 0 
futuredate 
futureDate��  ��   U  _ ` _ l  L S a���� a r   L S b c b m   L O d d � e e   c o      ���� 0 
exporttext 
exportText��  ��   `  f g f l     ��������  ��  ��   g  h i h l  T� j���� j O   T� k l k k   Z� m m  n o n l  Z a p q r p r   Z a s t s m   Z ] u u � v v  C a l e n d a r t o      ���� 0 calendarname calendarName q - ' Replace with the name of your calendar    r � w w N   R e p l a c e   w i t h   t h e   n a m e   o f   y o u r   c a l e n d a r o  x y x r   b n z { z 4   b j�� |
�� 
wres | o   f i���� 0 calendarname calendarName { o      ���� 0 thecalendar theCalendar y  } ~ } l  o o��������  ��  ��   ~   �  l  o o�� � ���   �   Get all events    � � � �    G e t   a l l   e v e n t s �  � � � r   o z � � � l  o v ����� � n   o v � � � 2   r v��
�� 
wrev � o   o r���� 0 thecalendar theCalendar��  ��   � o      ���� 0 	theevents 	theEvents �  � � � l  { {��������  ��  ��   �  ��� � X   {� ��� � � k   �� � �  � � � r   � � � � � l  � � ����� � n   � � � � � 1   � ���
�� 
wr1s � o   � ����� 0 theevent theEvent��  ��   � o      ����  0 eventstartdate eventStartDate �  � � � r   � � � � � l  � � ����� � n   � � � � � 1   � ���
�� 
wr5s � o   � ����� 0 theevent theEvent��  ��   � o      ���� 0 eventenddate eventEndDate �  � � � r   � � � � � l  � � ����� � n   � � � � � 1   � ���
�� 
wr4s � o   � ����� 0 theevent theEvent��  ��   � o      ����  0 eventstampdate eventStampDate �  � � � r   � � � � � m   � ���
�� boovfals � o      ���� 0 includeevent includeEvent �  � � � l  � ���������  ��  ��   �  � � � l  � ��� � ���   � D > Filter by DTSTAMP date range (last 365 days to next 365 days)    � � � � |   F i l t e r   b y   D T S T A M P   d a t e   r a n g e   ( l a s t   3 6 5   d a y s   t o   n e x t   3 6 5   d a y s ) �  � � � Z   � � � ����� � F   � � � � � @   � � � � � o   � �����  0 eventstampdate eventStampDate � o   � ����� 0 pastdate pastDate � B   � � � � � o   � �����  0 eventstampdate eventStampDate � o   � ����� 0 
futuredate 
futureDate � r   � � � � � m   � ���
�� boovtrue � o      ���� 0 includeevent includeEvent��  ��   �  � � � l  � ���������  ��  ��   �  � � � Z   �� � ����� � o   � ����� 0 includeevent includeEvent � k   �� � �  � � � r   � � � � � l  � � ����� � c   � � � � � n   � � � � � 1   � ��
� 
ID   � o   � ��~�~ 0 theevent theEvent � m   � ��}
�} 
TEXT��  ��   � o      �|�| 0 eventuid eventUID �  � � � r   � � � � � l  � � ��{�z � c   � � � � � n   � � � � � 1   � ��y
�y 
wr11 � o   � ��x�x 0 theevent theEvent � m   � ��w
�w 
TEXT�{  �z   � o      �v�v 0 eventsummary eventSummary �  � � � r   � � � � c   � � � � n   �  � � � 2   � �u
�u 
cha  � l  � � ��t�s � c   � � � � � n   � � � � � 1   � ��r
�r 
wr14 � o   � ��q�q 0 theevent theEvent � m   � ��p
�p 
TEXT�t  �s   � m   �o
�o 
TEXT � o      �n�n 0 eventlocation eventLocation �  � � � r   � � � c   � � � n   � � � 2  �m
�m 
cha  � l  ��l�k � c   � � � n   � � � 1  �j
�j 
wr12 � o  �i�i 0 theevent theEvent � m  �h
�h 
TEXT�l  �k   � m  �g
�g 
TEXT � o      �f�f $0 eventdescription eventDescription �  � � � r  $ � � � l   ��e�d � c    � � � n   � � � 1  �c
�c 
wr15 � o  �b�b 0 theevent theEvent � m  �a
�a 
TEXT�e  �d   � o      �`�` "0 eventrecurrence eventRecurrence �  � � � r  %0 � � � l %, ��_�^ � c  %, � � � n  %*   1  &*�]
�] 
wr16 o  %&�\�\ 0 theevent theEvent � m  *+�[
�[ 
TEXT�_  �^   � o      �Z�Z 0 eventurl eventURL �  r  1: l 16�Y�X n  16 1  26�W
�W 
wrad o  12�V�V 0 theevent theEvent�Y  �X   o      �U�U 0 eventallday eventAllDay 	
	 r  ;D l ;@�T�S n  ;@ 1  <@�R
�R 
wre4 o  ;<�Q�Q 0 theevent theEvent�T  �S   o      �P�P 0 eventstatus eventStatus
  r  EN l EJ�O�N n  EJ 1  FJ�M
�M 
wr4s o  EF�L�L 0 theevent theEvent�O  �N   o      �K�K 0 eventmodified eventModified  r  OX l OT�J�I n  OT 1  PT�H
�H 
wr13 o  OP�G�G 0 theevent theEvent�J  �I   o      �F�F 0 eventsequence eventSequence  r  Yb !  l Y^"�E�D" n  Y^#$# 2 Z^�C
�C 
wrea$ o  YZ�B�B 0 theevent theEvent�E  �D  ! o      �A�A  0 eventattendees eventAttendees %&% l cc�@�?�>�@  �?  �>  & '(' l cc�=)*�=  )   location cleaning   * �++ $   l o c a t i o n   c l e a n i n g( ,-, r  cx./. l ct0�<�;0 I ct�:1�9
�: .sysoexecTEXT���     TEXT1 b  cp232 b  cl454 m  cf66 �77 
 e c h o  5 n  fk898 1  ik�8
�8 
strq9 o  fi�7�7 0 eventlocation eventLocation3 m  lo:: �;; `   |   s e d   ' : a ; N ; $ ! b a ; s / \ n / - - - / g '   |   s e d   ' s / \ r / - - - / g '�9  �<  �;  / o      �6�6 0 cleanlocation cleanLocation- <=< r  y�>?> l y�@�5�4@ I y��3A�2
�3 .sysoexecTEXT���     TEXTA b  y�BCB m  y|DD �EE  t r   - d   ' 
 
 '   < < <  C n  |�FGF 1  ��1
�1 
strqG o  |�0�0 0 cleanlocation cleanLocation�2  �5  �4  ? o      �/�/ 0 cleanlocation cleanLocation= HIH r  ��JKJ l ��L�.�-L I ���,M�+
�, .sysoexecTEXT���     TEXTM b  ��NON b  ��PQP m  ��RR �SS 
 e c h o  Q n  ��TUT 1  ���*
�* 
strqU o  ���)�) 0 cleanlocation cleanLocationO m  ��VV �WW    |   t r   - d   ' 
 '�+  �.  �-  K o      �(�( 0 cleanlocation cleanLocationI XYX l ���'Z[�'  Z   description cleaning   [ �\\ *   d e s c r i p t i o n   c l e a n i n gY ]^] r  ��_`_ l ��a�&�%a I ���$b�#
�$ .sysoexecTEXT���     TEXTb b  ��cdc b  ��efe m  ��gg �hh 
 e c h o  f n  ��iji 1  ���"
�" 
strqj o  ���!�! $0 eventdescription eventDescriptiond m  ��kk �ll `   |   s e d   ' : a ; N ; $ ! b a ; s / \ n / - - - / g '   |   s e d   ' s / \ r / - - - / g '�#  �&  �%  ` o      � �  $0 cleandescription cleanDescription^ mnm r  ��opo l ��q��q I ���r�
� .sysoexecTEXT���     TEXTr b  ��sts m  ��uu �vv  t r   - d   ' 
 
 '   < < <  t n  ��wxw 1  ���
� 
strqx o  ���� $0 cleandescription cleanDescription�  �  �  p o      �� $0 cleandescription cleanDescriptionn yzy l ������  �  �  z {|{ r  ��}~} l ���� I �����
� .sysoexecTEXT���     TEXT� b  ����� b  ����� m  ���� ��� 
 e c h o  � n  ����� 1  ���
� 
strq� o  ���� $0 cleandescription cleanDescription� m  ���� ���    |   t r   - d   ' 
 '�  �  �  ~ o      �� $0 cleandescription cleanDescription| ��� l ������  �  �  � ��� l ������  �      � ���   � ��
� Z  �����	�� > ����� o  ���� 0 eventstatus eventStatus� m  ���� ���  C A N C E L L E D� k  ���� ��� l ������  �  �  � ��� r  ����� b  ����� b  ����� o  ���� 0 
exporttext 
exportText� m  ���� ���  B E G I N : V E V E N T� 1  ���
� 
lnfd� o      �� 0 
exporttext 
exportText� ��� r  ���� b  ���� b  ���� b  � ��� o  ��� �  0 
exporttext 
exportText� m  ���� ���  U I D :� o   ���� 0 eventuid eventUID� 1  ��
�� 
lnfd� o      ���� 0 
exporttext 
exportText� ��� r   ��� b  ��� b  ��� b  ��� o  ���� 0 
exporttext 
exportText� m  �� ���  S U M M A R Y :� o  ���� 0 eventsummary eventSummary� 1  ��
�� 
lnfd� o      ���� 0 
exporttext 
exportText� ��� Z  !������� o  !$���� 0 eventallday eventAllDay� k  'p�� ��� r  'H��� b  'D��� b  '@��� b  '.��� o  '*���� 0 
exporttext 
exportText� m  *-�� ��� & D T S T A R T ; V A L U E = D A T E :� l .?������ I .?�����
�� .sysoexecTEXT���     TEXT� b  .;��� b  .7��� m  .1�� ��� 4 d a t e   - j f   ' % A ,   % B   % d ,   % Y '   '� l 16������ c  16��� o  14����  0 eventstartdate eventStartDate� m  45��
�� 
TEXT��  ��  � m  7:�� ���  '   + ' % Y % m % d '��  ��  ��  � 1  @C��
�� 
lnfd� o      ���� 0 
exporttext 
exportText� ���� r  Ip��� b  Il��� b  Ih��� b  IP��� o  IL���� 0 
exporttext 
exportText� m  LO�� ��� " D T E N D ; V A L U E = D A T E :� l Pg������ I Pg�����
�� .sysoexecTEXT���     TEXT� b  Pc��� b  P_��� m  PS�� ��� 4 d a t e   - j f   ' % A ,   % B   % d ,   % Y '   '� l S^������ c  S^��� [  S\��� o  SV���� 0 eventenddate eventEndDate� ]  V[��� m  VW���� � 1  WZ��
�� 
days� m  \]��
�� 
TEXT��  ��  � m  _b�� ���  '   + ' % Y % m % d '��  ��  ��  � 1  hk��
�� 
lnfd� o      ���� 0 
exporttext 
exportText��  ��  � k  s��� ��� r  s���� b  s���� b  s�   b  sz o  sv���� 0 
exporttext 
exportText m  vy �  D T S T A R T : l z����� I z�����
�� .sysoexecTEXT���     TEXT b  z�	 b  z�

 m  z} � R d a t e   - u   - j f   ' % A ,   % B   % d ,   % Y   a t   % H : % M : % S '   ' l }����� c  }� o  }�����  0 eventstartdate eventStartDate m  ����
�� 
TEXT��  ��  	 m  �� � & '   + ' % Y % m % d T % H % M % S Z '��  ��  ��  � 1  ����
�� 
lnfd� o      ���� 0 
exporttext 
exportText� �� r  �� b  �� b  �� b  �� o  ������ 0 
exporttext 
exportText m  �� �  D T E N D : l ������ I ������
�� .sysoexecTEXT���     TEXT b  �� !  b  ��"#" m  ��$$ �%% R d a t e   - u   - j f   ' % A ,   % B   % d ,   % Y   a t   % H : % M : % S '   '# l ��&����& c  ��'(' o  ������ 0 eventenddate eventEndDate( m  ����
�� 
TEXT��  ��  ! m  ��)) �** & '   + ' % Y % m % d T % H % M % S Z '��  ��  ��   1  ����
�� 
lnfd o      ���� 0 
exporttext 
exportText��  � +,+ Z  ��-.����- F  ��/0/ F  ��121 > ��343 o  ������ 0 eventlocation eventLocation4 m  ��55 �66  2 > ��787 o  ������ 0 eventlocation eventLocation8 m  ��99 �::  m i s s i n g   v a l u e0 > ��;<; o  ������ 0 eventlocation eventLocation< m  ����
�� 
msng. r  ��=>= b  ��?@? b  ��ABA b  ��CDC o  ������ 0 
exporttext 
exportTextD m  ��EE �FF  L O C A T I O N :B o  ������ 0 cleanlocation cleanLocation@ 1  ����
�� 
lnfd> o      ���� 0 
exporttext 
exportText��  ��  , GHG Z  �2IJ����I F  �KLK F  �
MNM > ��OPO o  ������ 0 eventurl eventURLP m  ��QQ �RR  N > �STS o  ����� 0 eventurl eventURLT m  UU �VV  m i s s i n g   v a l u eL > WXW o  ���� 0 eventurl eventURLX m  ��
�� 
msngJ r  .YZY b  *[\[ b  &]^] b  "_`_ o  ���� 0 
exporttext 
exportText` m  !aa �bb  U R L :^ o  "%���� 0 eventurl eventURL\ 1  &)��
�� 
lnfdZ o      ���� 0 
exporttext 
exportText��  ��  H cdc Z  3pef����e F  3Vghg F  3Hiji > 3:klk o  36���� $0 eventdescription eventDescriptionl m  69mm �nn  j > =Dopo o  =@���� $0 eventdescription eventDescriptionp m  @Cqq �rr  m i s s i n g   v a l u eh > KRsts o  KN���� $0 eventdescription eventDescriptiont m  NQ��
�� 
msngf r  Yluvu b  Yhwxw b  Ydyzy b  Y`{|{ o  Y\���� 0 
exporttext 
exportText| m  \_}} �~~  D E S C R I P T I O N :z o  `c���� $0 cleandescription cleanDescriptionx 1  dg��
�� 
lnfdv o      ���� 0 
exporttext 
exportText��  ��  d � Z  q�������� F  q���� F  q���� > qx��� o  qt���� 0 eventmodified eventModified� m  tw�� ���  � > {���� o  {~���� 0 eventmodified eventModified� m  ~��� ���  m i s s i n g   v a l u e� > ����� o  ������ 0 eventmodified eventModified� m  ����
�� 
msng� r  ����� b  ����� b  ����� b  ����� o  ������ 0 
exporttext 
exportText� m  ���� ���  L A S T - M O D I F I E D :� l �������� I �������
�� .sysoexecTEXT���     TEXT� b  ����� b  ����� m  ���� ��� R d a t e   - u   - j f   ' % A ,   % B   % d ,   % Y   a t   % H : % M : % S '   '� l �������� c  ����� o  ������ 0 eventmodified eventModified� m  ����
�� 
TEXT��  ��  � m  ���� ��� & '   + ' % Y % m % d T % H % M % S Z '��  ��  ��  � 1  ����
�� 
lnfd� o      ���� 0 
exporttext 
exportText��  ��  � ��� Z  ��������� F  ����� F  ����� > ����� o  ������ 0 eventsequence eventSequence� m  ���� ���  � > ����� o  ������ 0 eventsequence eventSequence� m  ���� ���  m i s s i n g   v a l u e� > ����� o  ������ 0 eventsequence eventSequence� m  ����
�� 
msng� r  ����� b  ����� b  ����� b  ����� o  ������ 0 
exporttext 
exportText� m  ���� ���  S E Q U E N C E :� o  ������ 0 eventsequence eventSequence� 1  ����
�� 
lnfd� o      ���� 0 
exporttext 
exportText��  ��  � ��� l ����������  ��  ��  � ��� l ����������  ��  ��  � ��� r  ���� m  ���� ���  � o      ���� 0 attendeestext attendeesText� ��� X  ������ k  ��� ��� r  $��� l  ������ c   ��� n  ��� 1  ��
�� 
wra1� o  ���� 0 theattendee theAttendee� m  ��
�� 
TEXT��  ��  � o      ���� 0 attendeename attendeeName� ��� r  %0��� l %,���~� c  %,��� n  %*��� 1  &*�}
�} 
wra2� o  %&�|�| 0 theattendee theAttendee� m  *+�{
�{ 
TEXT�  �~  � o      �z�z 0 attendeeemail attendeeEmail� ��� r  1<��� l 18��y�x� c  18��� n  16��� 1  26�w
�w 
wra3� o  12�v�v 0 theattendee theAttendee� m  67�u
�u 
TEXT�y  �x  � o      �t�t  0 attendeestatus attendeeStatus� ��� Z  =Y���s�r� E =Q��� J  =M�� ��� m  =@�� ���  N o n e� ��� m  @C�� ���  T E N T A T I V E� ��� m  CF�� �    C O N F I R M E D� �q m  FI �  C A N C E L L E D�q  � o  MP�p�p  0 attendeestatus attendeeStatus� l TT�o�o   2 , Add attendeeStatus to required output here.    � X   A d d   a t t e n d e e S t a t u s   t o   r e q u i r e d   o u t p u t   h e r e .�s  �r  �  l ZZ�n�m�l�n  �m  �l   	
	 l ZZ�k�j�i�k  �j  �i  
  l ZZ�h�h   !  Clean the data like before    � 6   C l e a n   t h e   d a t a   l i k e   b e f o r e  r  Zk l Zg�g�f I Zg�e�d
�e .sysoexecTEXT���     TEXT b  Zc m  Z] �  t r   - d   ' 
 
 '   < < <   n  ]b 1  `b�c
�c 
strq o  ]`�b�b 0 attendeename attendeeName�d  �g  �f   o      �a�a 0 	cleanname 	cleanName  r  l} l ly �`�_  I ly�^!�]
�^ .sysoexecTEXT���     TEXT! b  lu"#" m  lo$$ �%%  t r   - d   ' 
 
 '   < < <  # n  ot&'& 1  rt�\
�\ 
strq' o  or�[�[ 0 attendeeemail attendeeEmail�]  �`  �_   o      �Z�Z 0 
cleanemail 
cleanEmail ()( l ~~�Y�X�W�Y  �X  �W  ) *+* l ~~�V,-�V  , p j Construct attendee data line in ICS format, though there's no standard attribute for participation status   - �.. �   C o n s t r u c t   a t t e n d e e   d a t a   l i n e   i n   I C S   f o r m a t ,   t h o u g h   t h e r e ' s   n o   s t a n d a r d   a t t r i b u t e   f o r   p a r t i c i p a t i o n   s t a t u s+ /�U/ r  ~�010 b  ~�232 b  ~�454 b  ~�676 b  ~�898 b  ~�:;: o  ~��T�T 0 attendeestext attendeesText; m  ��<< �==  A T T E N D E E ; C N =9 o  ���S�S 0 	cleanname 	cleanName7 m  ��>> �??  : M A I L T O :5 o  ���R�R 0 
cleanemail 
cleanEmail3 1  ���Q
�Q 
lnfd1 o      �P�P 0 attendeestext attendeesText�U  �� 0 theattendee theAttendee� o  	�O�O  0 eventattendees eventAttendees� @A@ Z ��BC�N�MB F  ��DED F  ��FGF > ��HIH o  ���L�L 0 attendeestext attendeesTextI m  ��JJ �KK  G > ��LML o  ���K�K 0 attendeestext attendeesTextM m  ��NN �OO  m i s s i n g   v a l u eE > ��PQP o  ���J�J 0 attendeestext attendeesTextQ m  ���I
�I 
msngC r  ��RSR b  ��TUT o  ���H�H 0 
exporttext 
exportTextU o  ���G�G 0 attendeestext attendeesTextS o      �F�F 0 
exporttext 
exportText�N  �M  A V�EV r  ��WXW b  ��YZY b  ��[\[ o  ���D�D 0 
exporttext 
exportText\ m  ��]] �^^  E N D : V E V E N TZ 1  ���C
�C 
lnfdX o      �B�B 0 
exporttext 
exportText�E  �	  �  �
  ��  ��   � _�A_ I ���@`�?
�@ .sysodelanull��� ��� nmbr` m  ��aa ?��������?  �A  �� 0 theevent theEvent � o   ~ ��>�> 0 	theevents 	theEvents��   l m   T Wbb�                                                                                  wrbt  alis    &  2TB                        �_�BD ����Calendar.app                                                   �����_�        ����  
 cu             Applications  #/:System:Applications:Calendar.app/     C a l e n d a r . a p p    2 T B   System/Applications/Calendar.app  / ��  ��  ��   i cdc l     �=�<�;�=  �<  �;  d efe l     �:gh�:  g    Add the final ICS content   h �ii 4   A d d   t h e   f i n a l   I C S   c o n t e n tf jkj l �l�9�8l r  �mnm b  �opo m  ��qq �rr  B E G I N : V C A L E N D A Rp 1  ��7
�7 
lnfdn o      �6�6 0 
icscontent 
icsContent�9  �8  k sts l u�5�4u r  vwv b  xyx b  z{z o  
�3�3 0 
icscontent 
icsContent{ m  
|| �}}  V E R S I O N : 2 . 0y 1  �2
�2 
lnfdw o      �1�1 0 
icscontent 
icsContent�5  �4  t ~~ l &��0�/� r  &��� b  "��� b  ��� o  �.�. 0 
icscontent 
icsContent� m  �� ��� 2 P R O D I D : - / / s i j / / s i j a p i / / E N� 1  !�-
�- 
lnfd� o      �,�, 0 
icscontent 
icsContent�0  �/   ��� l '2��+�*� r  '2��� b  '.��� o  '*�)�) 0 
icscontent 
icsContent� o  *-�(�( 0 
exporttext 
exportText� o      �'�' 0 
icscontent 
icsContent�+  �*  � ��� l 3B��&�%� r  3B��� b  3>��� b  3:��� o  36�$�$ 0 
icscontent 
icsContent� m  69�� ���  E N D : V C A L E N D A R� 1  :=�#
�# 
lnfd� o      �"�" 0 
icscontent 
icsContent�&  �%  � ��� l     �!� ��!  �   �  � ��� l     ����  � 2 , Write the ICS content to the temporary file   � ��� X   W r i t e   t h e   I C S   c o n t e n t   t o   t h e   t e m p o r a r y   f i l e� ��� l CX���� I CX���
� .sysoexecTEXT���     TEXT� b  CT��� b  CP��� b  CL��� m  CF�� ��� 
 e c h o  � n  FK��� 1  IK�
� 
strq� o  FI�� 0 
icscontent 
icsContent� m  LO�� ���    >  � n  PS��� 1  QS�
� 
strq� o  PQ��  0 tempexportfile tempExportFile�  �  �  � ��� l     ����  �  �  � ��� l     ����  � 6 0 Move the temporary file to the desired location   � ��� `   M o v e   t h e   t e m p o r a r y   f i l e   t o   t h e   d e s i r e d   l o c a t i o n� ��� l Yl���� I Yl���
� .sysoexecTEXT���     TEXT� b  Yh��� b  Yd��� b  Y`��� m  Y\�� ���  m v  � n  \_��� 1  ]_�
� 
strq� o  \]��  0 tempexportfile tempExportFile� m  `c�� ���   � n  dg��� 1  eg�
� 
strq� o  de�
�
 0 
exportfile 
exportFile�  �  �  � ��� l     �	���	  �  �  � ��� l     ����  � ; 5 display dialog "Exported calendar to: " & exportFile   � ��� j   d i s p l a y   d i a l o g   " E x p o r t e d   c a l e n d a r   t o :   "   &   e x p o r t F i l e� ��� l     ����  �  �  �       B����������� u������ ���������������������������������������������������������������������������������  � @��������������������������������������������������������������������������������������������������������������������������������
�� .aevtoappnull  �   � ****�� 0 
homefolder 
homeFolder�� 0 exportfolder exportFolder�� 0 
exportfile 
exportFile��  0 tempexportfile tempExportFile�� 	0 today  �� 0 pastdate pastDate�� 0 
futuredate 
futureDate�� 0 
exporttext 
exportText�� 0 calendarname calendarName�� 0 thecalendar theCalendar�� 0 	theevents 	theEvents��  0 eventstartdate eventStartDate�� 0 eventenddate eventEndDate��  0 eventstampdate eventStampDate�� 0 includeevent includeEvent�� 0 eventuid eventUID�� 0 eventsummary eventSummary�� 0 eventlocation eventLocation�� $0 eventdescription eventDescription�� "0 eventrecurrence eventRecurrence�� 0 eventurl eventURL�� 0 eventallday eventAllDay�� 0 eventstatus eventStatus�� 0 eventmodified eventModified�� 0 eventsequence eventSequence��  0 eventattendees eventAttendees�� 0 cleanlocation cleanLocation�� $0 cleandescription cleanDescription�� 0 attendeestext attendeesText�� 0 attendeename attendeeName�� 0 attendeeemail attendeeEmail��  0 attendeestatus attendeeStatus�� 0 	cleanname 	cleanName�� 0 
cleanemail 
cleanEmail��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  � �����������
�� .aevtoappnull  �   � ****� k    l��  ��  ��  ��  #��  3��  D��  I��  T��  _��  h   j s ~ � � � �����  ��  ��  � ������ 0 theevent theEvent�� 0 theattendee theAttendee� ����������� �� !�� *�� 9������������������ d��b u����������������~�}�|�{�z�y�x�w�v�u�t�s�r�q�p�o�n�m�l�k�j�i�h�g�f�e�d�c�b�a6:�`DRVgk�_u�����^��������$)59�]EQUamq}����������\�[�Z�Y�X�W�V����U�T$�S<>JN]a�Rq�Q|������
�� afdrcusr
�� .earsffdralis        afdr
�� 
psxp
�� 
TEXT�� 0 
homefolder 
homeFolder�� 0 exportfolder exportFolder�� 0 
exportfile 
exportFile��  0 tempexportfile tempExportFile
�� 
strq
�� .sysoexecTEXT���     TEXT
�� .misccurdldt    ��� null�� 	0 today  �� 
�� 
days�� 0 pastdate pastDate�� Z�� 0 
futuredate 
futureDate�� 0 
exporttext 
exportText�� 0 calendarname calendarName
�� 
wres�� 0 thecalendar theCalendar
�� 
wrev�� 0 	theevents 	theEvents
�� 
kocl
�� 
cobj
� .corecnte****       ****
�~ 
wr1s�}  0 eventstartdate eventStartDate
�| 
wr5s�{ 0 eventenddate eventEndDate
�z 
wr4s�y  0 eventstampdate eventStampDate�x 0 includeevent includeEvent
�w 
bool
�v 
ID  �u 0 eventuid eventUID
�t 
wr11�s 0 eventsummary eventSummary
�r 
wr14
�q 
cha �p 0 eventlocation eventLocation
�o 
wr12�n $0 eventdescription eventDescription
�m 
wr15�l "0 eventrecurrence eventRecurrence
�k 
wr16�j 0 eventurl eventURL
�i 
wrad�h 0 eventallday eventAllDay
�g 
wre4�f 0 eventstatus eventStatus�e 0 eventmodified eventModified
�d 
wr13�c 0 eventsequence eventSequence
�b 
wrea�a  0 eventattendees eventAttendees�` 0 cleanlocation cleanLocation�_ $0 cleandescription cleanDescription
�^ 
lnfd
�] 
msng�\ 0 attendeestext attendeesText
�[ 
wra1�Z 0 attendeename attendeeName
�Y 
wra2�X 0 attendeeemail attendeeEmail
�W 
wra3�V  0 attendeestatus attendeeStatus�U �T 0 	cleanname 	cleanName�S 0 
cleanemail 
cleanEmail
�R .sysodelanull��� ��� nmbr�Q 0 
icscontent 
icsContent��m�j �,�&E�O��%E�O��%E�O��%E�O���,%j O*j E�O�a _  E` O�a _  E` Oa E` Oa �a E` O*a _ /E` O_ a -E` O}_ [a a l  kh  �a !,E` "O�a #,E` $O�a %,E` &OfE` 'O_ &_ 	 _ &_ a (& 
eE` 'Y hO_ '�a ),�&E` *O�a +,�&E` ,O�a -,�&a .-�&E` /O�a 0,�&a .-�&E` 1O�a 2,�&E` 3O�a 4,�&E` 5O�a 6,E` 7O�a 8,E` 9O�a %,E` :O�a ;,E` <O�a =-E` >Oa ?_ /�,%a @%j E` AOa B_ A�,%j E` AOa C_ A�,%a D%j E` AOa E_ 1�,%a F%j E` GOa H_ G�,%j E` GOa I_ G�,%a J%j E` GO_ 9a K _ a L%_ M%E` O_ a N%_ *%_ M%E` O_ a O%_ ,%_ M%E` O_ 7 N_ a P%a Q_ "�&%a R%j %_ M%E` O_ a S%a T_ $k_  �&%a U%j %_ M%E` Y E_ a V%a W_ "�&%a X%j %_ M%E` O_ a Y%a Z_ $�&%a [%j %_ M%E` O_ /a \	 _ /a ]a (&	 _ /a ^a (& _ a _%_ A%_ M%E` Y hO_ 5a `	 _ 5a aa (&	 _ 5a ^a (& _ a b%_ 5%_ M%E` Y hO_ 1a c	 _ 1a da (&	 _ 1a ^a (& _ a e%_ G%_ M%E` Y hO_ :a f	 _ :a ga (&	 _ :a ^a (& &_ a h%a i_ :�&%a j%j %_ M%E` Y hO_ <a k	 _ <a la (&	 _ <a ^a (& _ a m%_ <%_ M%E` Y hOa nE` oO �_ >[a a l  kh �a p,�&E` qO�a r,�&E` sO�a t,�&E` uOa va wa xa ya zv_ u hY hOa {_ q�,%j E` |Oa }_ s�,%j E` ~O_ oa %_ |%a �%_ ~%_ M%E` o[OY�zO_ oa �	 _ oa �a (&	 _ oa ^a (& _ _ o%E` Y hO_ a �%_ M%E` Y hY hOa �j �[OY��UOa �_ M%E` �O_ �a �%_ M%E` �O_ �a �%_ M%E` �O_ �_ %E` �O_ �a �%_ M%E` �Oa �_ ��,%a �%��,%j Oa ���,%a �%��,%j � �  / U s e r s / s i j /� � @ / U s e r s / s i j / w o r k s h o p / s i j a p i / d a t a /� �		 X / U s e r s / s i j / w o r k s h o p / s i j a p i / d a t a / c a l e n d a r . i c s� �

 8 / U s e r s / s i j / c a l e n d a r _ t e m p . i c s� ldt     �h� ldt     �s�� ldt     ��� �.� B E G I N : V E V E N T 
 U I D : A 8 0 F 9 8 9 A - F F 1 6 - 4 1 A D - B 9 D 8 - D 2 D 6 9 0 0 5 E 4 F F 
 S U M M A R Y : S u m m o n e d   f o r   j u r y   s e l e c t i o n . 
 D T S T A R T : 2 0 2 3 1 1 0 1 T 0 8 0 0 0 0 Z 
 D T E N D : 2 0 2 3 1 1 0 1 T 1 2 0 0 0 0 Z 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 5 1 3 Z 
 S E Q U E N C E : 0 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 0 8 9 3 F E E 4 - 5 D C 4 - 4 D 2 A - B 1 9 8 - 1 C 2 5 2 B 3 9 B 5 C 2 
 S U M M A R Y : S a n g y e   O O O 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 1 2 2 6 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 1 2 3 0 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 5 0 8 Z 
 S E Q U E N C E : 1 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : D C 4 D 8 4 C 2 - E F A D - 4 0 9 A - B 8 2 E - 8 F 3 0 3 7 E D 0 7 8 3 
 S U M M A R Y : S a n g y e   O O O 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 1 2 1 8 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 1 2 2 3 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 5 0 3 Z 
 S E Q U E N C E : 1 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 0 8 D C C E 5 A - A F 2 E - 4 3 5 1 - A D D E - 9 F 1 A 6 B E A E 9 E 3 
 S U M M A R Y : S a n g y e   O O O 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 1 2 1 4 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 1 2 1 6 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 5 0 3 Z 
 S E Q U E N C E : 1 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 6 1 D 8 E 4 F 8 - 2 F 5 4 - 4 E 7 E - 9 C 6 2 - 8 3 D F 3 2 0 A D 1 9 9 
 S U M M A R Y : S a n g y e   o u t   o f   t h e   o f f i c e 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 1 0 0 4 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 1 0 0 7 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 5 0 8 Z 
 S E Q U E N C E : 1 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : C 1 D 6 C 3 F A - F 5 A C - 4 E 6 C - 9 1 0 F - 5 4 9 2 2 D 3 2 1 C B 6 
 S U M M A R Y : D e a d l i n e   t o   f i l e   N 1 2 6   a m e n d e d   c o m p l a i n t 
 D T S T A R T : 2 0 2 3 0 9 0 7 T 0 9 0 0 0 0 Z 
 D T E N D : 2 0 2 3 0 9 0 7 T 1 0 0 0 0 0 Z 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 4 5 2 Z 
 S E Q U E N C E : 0 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 2 7 E 4 3 A B F - 5 2 7 0 - 4 4 B F - 9 D 8 F - 8 8 5 C B 4 0 6 0 5 3 B 
 S U M M A R Y : U   o f   O   c l i n i c 
 D T S T A R T : 2 0 2 3 0 8 2 3 T 0 8 4 5 0 0 Z 
 D T E N D : 2 0 2 3 0 8 2 3 T 0 9 5 0 0 0 Z 
 L O C A T I O N : 1 5 1 5   A g a t e   S t  E u g e n e ,   O R ,   U n i t e d   S t a t e s 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 4 5 2 Z 
 S E Q U E N C E : 4 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 7 1 1 5 3 9 F B - 7 B 8 D - 4 6 3 D - A 3 D 9 - F E 8 8 C 2 3 F C F 7 F 
 S U M M A R Y : O R   T M D L s   c a l l 
 D T S T A R T : 2 0 2 3 0 8 0 2 T 1 1 0 0 0 0 Z 
 D T E N D : 2 0 2 3 0 8 0 2 T 1 2 0 0 0 0 Z 
 D E S C R I P T I O N : W e   c a n   u s e   o u r   c o n f e r e n c e   l i n e :    �  D i a l   ( 7 2 0 )   7 4 0 - 9 7 3 1  A c c e s s   c o d e :   3 0 6 4 4 2 8 #   H o s t   P I N :   7 5 8 9 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 4 2 0 Z 
 S E Q U E N C E : 1 
 A T T E N D E E ; C N = J a m e s   S a u l : M A I L T O : j s a u l @ l c l a r k . e d u 
 A T T E N D E E ; C N = n b e l l @ a d v o c a t e s - n w e a . o r g : M A I L T O : n b e l l @ a d v o c a t e s - n w e a . o r g 
 A T T E N D E E ; C N = S a n g y e   I n c e - J o h a n n s e n : M A I L T O : s a n g y e i j @ w e s t e r n l a w . o r g 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : A 0 7 D 1 F 2 5 - 9 7 F 7 - 4 B C D - 8 A F 6 - 9 9 5 B 8 6 8 F 2 1 0 9 
 S U M M A R Y : N E D C   M e e t i n g 
 D T S T A R T : 2 0 1 7 0 8 3 0 T 1 2 1 0 0 0 Z 
 D T E N D : 2 0 1 7 0 8 3 0 T 1 3 1 0 0 0 Z 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 2 9 4 3 Z 
 S E Q U E N C E : 1 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 9 D 1 6 4 E A C - 2 5 0 E - 4 A 4 3 - B 2 8 0 - 7 7 A 6 2 E B C D A 1 0 
 S U M M A R Y : R e h e a r s a l 
 D T S T A R T : 2 0 1 7 0 8 1 9 T 1 3 0 0 0 0 Z 
 D T E N D : 2 0 1 7 0 8 1 9 T 1 4 0 0 0 0 Z 
 L O C A T I O N : S i l k   R o a d   S t a g e 
 D E S C R I P T I O N : S a t   A u g   1 9           S i l k   R o a d   S t a g e  1   p m     R e h e a r s a l ,   " T a l i s m a n   C a r a v a n "  M e e t i n g   O n l y .   N o   c a m e r a   n e e d e d . 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 2 6 Z 
 S E Q U E N C E : 0 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : C D 0 1 9 7 C 4 - 8 9 E 5 - 4 8 D 9 - B 6 5 A - 9 0 A 7 1 1 8 A 1 D A 0 
 S U M M A R Y : C l i m a t e   C h a n g e   a t   t h e   B a r   a n d   B e n c h :   A   P r e s e n t a t i o n   o f   t h e   E L I   C l i m a t e   J u d i c i a r y   P r o j e c t 
 D T S T A R T : 2 0 2 2 0 3 2 2 T 0 9 3 0 0 0 Z 
 D T E N D : 2 0 2 2 0 3 2 2 T 1 0 3 0 0 0 Z 
 L O C A T I O N : W a s h i n g t o n ,   D C 
 U R L : f b : / / e v e n t / ? i d = 1 3 0 4 3 9 9 4 0 0 0 6 9 1 0 6 
 D E S C R I P T I O N : T h e   o v e r   1 2 0 0   c l i m a t e   l i t i g a t i o n   c a s e s   f i l e d   i n   t h e   U n i t e d   S t a t e s ,   r e f l e c t s   a n   e m e r g i n g   c h a l l e n g e   t o   m e m b e r s   o f   t h e   b a r   a n d   b e n c h   a b o u t   c o n s i d e r i n g   i s s u e s   o f   c l i m a t e   s c i e n c e   a n d   i t s   i m p l i c a t i o n s   o n   q u e s t i o n s   o f   f e d e r a l ,   s t a t e ,   a n d   a d m i n i s t r a t i v e   l a w .   S i n c e   1 9 9 0 ,   t h e   E n v i r o n m e n t a l   L a w   I n s t i t u t e   ( E L I )   h a s   a s s i s t e d   m o r e   t h a n   3 0 0 0   j u d g e s   f r o m   2 8   c o u n t r i e s   w i t h   e d u c a t i o n   a n d   r e s o u r c e s   o n   a   w i d e   r a n g e   o f   t o p i c s .   F o u n d e d   i n   2 0 1 8 ,   t h e   C l i m a t e   J u d i c i a r y   P r o j e c t   p r o v i d e s   i n f o r m a t i o n   a n d   t r a i n i n g   f o r   t h e   b a r   a n d   b e n c h   t o   k e e p   p a c e   w i t h   t h e   c u r r e n t   c o n s e n s u s   o n   c l i m a t e   s c i e n c e   a n d   e m e r g i n g   d e v e l o p m e n t s   i n   t h e   l a w .   J o i n   u s   M a r c h   2 2   f o r   a   w e b i n a r   t h a t   w i l l   i n t r o d u c e   t h e   o b j e c t i v e s   a n d   c u r r i c u l a   o f   t h e   C l i m a t e   J u d i c i a r y   P r o j e c t   a n d   p r o v i d e   a n   o v e r v i e w   o f   t h e   t o p i c s   a n d   i s s u e s   a d d r e s s e d   i n   t h e   e d u c a t i o n a l   p r o g r a m s   a n d   m a t e r i a l s :   h t t p s : / / b i t . l y / 3 I b 0 I D u 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 2 0 Z 
 S E Q U E N C E : 0 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : C 9 9 7 9 C 7 4 - 1 A C A - 4 E 5 C - B C F 9 - 7 1 B 5 C 7 B 3 9 4 E 2 
 S U M M A R Y : C l e a n i n g   d a y 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 0 7 1 5 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 0 7 1 6 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 2 6 Z 
 S E Q U E N C E : 2 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 0 F A F C 9 3 7 - 1 0 6 C - 4 A C 3 - B 7 B 9 - 9 C 1 5 8 1 9 E 2 6 A 4 
 S U M M A R Y : O R   T M D L s   &   W A   W O L V E S 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 0 8 0 2 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 0 8 0 3 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 3 1 Z 
 S E Q U E N C E : 2 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : E 9 6 6 8 1 4 A - 3 B 2 B - 4 5 3 7 - B 4 6 4 - 3 3 5 F 9 2 4 F 5 9 9 6 
 S U M M A R Y : C l i n i c   F a i r 
 D T S T A R T : 2 0 2 3 0 2 2 7 T 1 2 0 0 0 0 Z 
 D T E N D : 2 0 2 3 0 2 2 7 T 1 3 0 0 0 0 Z 
 L O C A T I O N : U n i v e r s i t y   o f   O r e g o n   S c h o o l   o f   L a w  1 5 1 5   A g a t e   S t  E u g e n e   O R   9 7 4 0 3 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 3 1 Z 
 S E Q U E N C E : 0 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : E D 7 7 E E 6 7 - 3 9 8 F - 4 A 0 5 - 8 8 A D - 1 2 F A 4 B C 0 0 A C 1 
 S U M M A R Y : P l a n   m y   w e e k 
 D T S T A R T : 2 0 2 3 0 5 2 8 T 1 1 3 0 0 0 Z 
 D T E N D : 2 0 2 3 0 5 2 8 T 1 4 3 0 0 0 Z 
 L O C A T I O N : 3 5   C l u b   R o a d  A p a r t m e n t   4 0 3 ,   E u g e n e ,   U n i t e d   S t a t e s 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 3 6 Z 
 S E Q U E N C E : 1 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 9 6 1 9 3 B F 4 - 4 8 C 3 - 4 8 3 1 - A 5 B 6 - 6 2 2 F 4 8 C 1 D 0 E 2 
 S U M M A R Y : B o x i n g   a n d   s o m e w h a t - c l e a n i n g   d a y 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 0 7 1 3 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 0 7 1 4 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 3 6 Z 
 S E Q U E N C E : 2 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 8 4 6 B 1 6 A 6 - 6 7 B 5 - 4 7 1 9 - A 6 E B - C B D 9 3 C 1 2 E 0 7 4 
 S U M M A R Y : C a l l   w i t h   T i m   C o l e m a n 
 D T S T A R T : 2 0 2 3 0 7 1 0 T 1 4 0 0 0 0 Z 
 D T E N D : 2 0 2 3 0 7 1 0 T 1 5 0 0 0 0 Z 
 L O C A T I O N : K e t t l e   R a n g e 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 4 2 Z 
 S E Q U E N C E : 0 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : A 8 0 A 1 A 4 A - 9 B 3 3 - 4 F 4 7 - 9 2 1 8 - A 2 E 1 0 F 3 A F C 9 7 
 S U M M A R Y : W E L C   A n n u a l   S t a f f   &   B o a r d   R e t r e a t 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 0 9 1 2 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 0 9 1 5 
 L O C A T I O N : T o l o v a n a   I n n  3 4 0 0   S   H e m l o c k  C a n n o n   B e a c h   O R   9 7 1 4 5  U n i t e d   S t a t e s 
 U R L : r e a d d l e - s p a r k : / / b l = Q T p z Y W 5 n e W V p a k B 3 Z X N 0 Z X J u b G F 3 L m 9 y Z z t J R D o 0 Q T E x R E E 2 Q S 0 y N z B E L T R G M j I t % 0 D % 0 A Q k F C N i 0 3 Q k M 3 M D E w O E V G Q 0 Z A d 2 V z d G V y b m x h d y 5 v c m c 7 Z U l E O k F B T W t B R 0 p o T V R S % 0 D % 0 A b U 5 t U X d M V F l 3 W m p r d E 5 E U m p a a T F o W k R J M U x X U X d P R 1 Z r T k R N e l l t T m t P U U J H Q U F B % 0 D % 0 A Q U F B R E 5 u K 2 5 C Q 1 N F R F R x K 0 N l c 2 R R d k t k W U J 3 Q 2 F C U F R Q V m N a S l N v N j h k V V d i R X Z 1 % 0 D % 0 A d k F B Q U F B Q U V N Q U F D Y U J Q V F B W Y 1 p K U 2 8 2 O G R V V 2 J F d n V 2 Q U F C R 2 d a N G R B Q U E 9 O z M x % 0 D % 0 A M z Y 3 O D E 5 N z E % 3 D 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 4 4 Z 
 S E Q U E N C E : 4 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 3 5 F C F 5 7 A - 5 5 C 9 - 4 A F 9 - 9 D 0 F - 1 1 C 5 A 6 3 C 4 9 9 1 
 S U M M A R Y : P e t e   g o i n g   o v e r t o   S h a s t a 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 0 8 1 6 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 0 8 1 7 
 L O C A T I O N : S h a s t a 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 4 4 Z 
 S E Q U E N C E : 2 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 3 5 0 5 E 5 A 4 - 0 6 0 8 - 4 5 2 9 - B D 9 9 - 0 9 4 6 2 B 0 5 E 1 8 0 
 S U M M A R Y : S A L D F   M e e t i n g 
 D T S T A R T : 2 0 1 7 0 8 3 1 T 1 2 1 0 0 0 Z 
 D T E N D : 2 0 1 7 0 8 3 1 T 1 3 3 0 0 0 Z 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 2 9 5 1 Z 
 S E Q U E N C E : 2 
 E N D : V E V E N T 
 B E G I N : V E V E N T 
 U I D : 0 4 D D 0 2 4 5 - 7 6 0 F - 4 0 B 7 - 8 7 9 D - 7 1 F 3 9 1 A 2 0 F 7 2 
 S U M M A R Y : W A   W O L V E S   &   H O U R S 
 D T S T A R T ; V A L U E = D A T E : 2 0 2 3 0 8 0 3 
 D T E N D ; V A L U E = D A T E : 2 0 2 3 0 8 0 5 
 L A S T - M O D I F I E D : 2 0 2 4 0 5 2 9 T 1 0 4 7 3 6 Z 
 S E Q U E N C E : 2 
 E N D : V E V E N T 
�  b�P�O
�P 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�O kfrmID  � �N�N  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~�������������������������������������������������������������������������������������������������������������������������������� 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~�������������������������������������������������������������������������������������������������������������������������������� 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~�������������������������������������������������������������������������������������������������������������������������������� 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~�������������������������������������������������������������������������������������������������������������������������������� 	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~��������������������������������������������������������������������������������������������������������������������������������	 											
											 		 	�M	�L	 b�K	�J
�K 
wres	 �		 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev	 �		 H 5 7 7 5 C A 9 6 - D 4 5 E - 4 6 A 7 - 9 6 6 C - B D C 8 2 1 5 0 C 2 D 0
�L kfrmID   		 	�I	�H	 b�G	�F
�G 
wres	 �	 	  H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev	 �	!	! H 5 C A 1 2 0 7 8 - 0 F 6 4 - 4 B 1 3 - A 2 0 5 - 2 2 1 1 1 1 2 D 2 A F A
�H kfrmID   	"	" 	#�E	$�D	# b�C	%�B
�C 
wres	% �	&	& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev	$ �	'	' H 9 E 5 3 3 7 3 7 - B 7 B B - 4 0 B 4 - A 9 B 6 - 2 5 9 3 C C A A 1 0 9 6
�D kfrmID   	(	( 	)�A	*�@	) b�?	+�>
�? 
wres	+ �	,	, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev	* �	-	- H 5 2 7 6 7 7 E C - B B 7 B - 4 0 B D - B 5 1 F - 3 0 D E E F F F 9 8 8 6
�@ kfrmID   	.	. 	/�=	0�<	/ b�;	1�:
�; 
wres	1 �	2	2 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev	0 �	3	3 H 6 5 6 7 6 8 5 A - C F 0 3 - 4 7 2 E - 9 E 2 7 - D E D 7 9 F 8 4 3 F F A
�< kfrmID   	4	4 	5�9	6�8	5 b�7	7�6
�7 
wres	7 �	8	8 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev	6 �	9	9 H 6 6 A 4 8 0 F A - 6 4 8 2 - 4 8 4 5 - 8 4 0 4 - F D E 2 D C 7 1 8 C F E
�8 kfrmID   	:	: 	;�5	<�4	; b�3	=�2
�3 
wres	= �	>	> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev	< �	?	? H D E F 2 B 1 9 3 - 3 E 9 C - 4 8 5 8 - B 1 B 7 - C 4 0 E 2 A 0 F C 0 B 9
�4 kfrmID   	@	@ 	A�1	B�0	A b�/	C�.
�/ 
wres	C �	D	D H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev	B �	E	E H 7 3 6 9 D 8 A 6 - 4 C 4 B - 4 C F B - 8 4 F 1 - 0 D 3 4 A 0 2 3 A F 1 8
�0 kfrmID   	F	F 	G�-	H�,	G b�+	I�*
�+ 
wres	I �	J	J H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev	H �	K	K H E D 9 E A 6 3 6 - A 1 7 F - 4 9 F 6 - 9 8 4 7 - A 6 2 6 7 8 7 A 8 9 7 6
�, kfrmID   	L	L 	M�)	N�(	M b�'	O�&
�' 
wres	O �	P	P H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev	N �	Q	Q H A 2 A 1 2 A D 2 - 1 E 6 4 - 4 4 3 3 - 8 4 1 6 - 6 E E D 5 0 9 2 2 7 4 3
�( kfrmID   	R	R 	S�%	T�$	S b�#	U�"
�# 
wres	U �	V	V H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev	T �	W	W H D 1 7 D F A 5 0 - B 8 2 5 - 4 C A 9 - 8 D 9 C - 0 9 C A A 8 3 9 3 F A A
�$ kfrmID   	X	X 	Y�!	Z� 	Y b�	[�
� 
wres	[ �	\	\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev	Z �	]	] H 3 0 4 8 9 3 2 E - 2 5 B 2 - 4 B E 3 - B 0 4 9 - 1 9 A F 0 C 5 6 D E 2 5
�  kfrmID   	^	^ 	_�	`�	_ b�	a�
� 
wres	a �	b	b H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev	` �	c	c H E 7 B F 1 A 4 6 - F 1 7 9 - 4 D 8 A - 8 0 4 4 - D 8 8 D A F 2 1 4 A 9 A
� kfrmID   	d	d 	e�	f�	e b�	g�
� 
wres	g �	h	h H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev	f �	i	i H 7 7 7 3 D A 4 4 - 7 4 8 6 - 4 B 7 F - 9 9 3 1 - 8 B 6 7 E C 7 C C C 3 9
� kfrmID   	j	j 	k�	l�	k b�	m�
� 
wres	m �	n	n H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev	l �	o	o H E A 6 8 7 0 B F - 4 3 C E - 4 3 6 6 - B 8 7 0 - C 3 3 6 3 B 1 3 F D 2 6
� kfrmID   	p	p 	q�	r�	q b�	s�
� 
wres	s �	t	t H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev	r �	u	u H 4 9 2 D 2 A D C - 6 4 0 E - 4 F 8 1 - B 6 F C - 2 8 C 1 2 C 3 A 5 E 0 1
� kfrmID    	v	v 	w�	x�	w b�	y�

� 
wres	y �	z	z H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev	x �	{	{ H D 9 C A C 1 8 B - 6 B D 2 - 4 9 7 8 - B 3 D 6 - 5 9 F A 3 7 2 A B 2 4 A
� kfrmID  ! 	|	| 	}�		~�	} b�	�
� 
wres	 �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev	~ �	�	� H E 4 D 9 5 A E 3 - 6 7 3 3 - 4 3 6 6 - 9 5 8 1 - 3 7 F 8 7 B C 2 5 D 4 E
� kfrmID  " 	�	� 	��	��	� b�	��
� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev	� �	�	� H 1 8 F 2 A A 9 6 - 9 9 E 4 - 4 0 7 1 - 8 4 4 8 - 4 1 6 F 3 D 6 E F 4 5 7
� kfrmID  # 	�	� 	��	�� 	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev	� �	�	� H 8 8 F 4 D 8 1 2 - 9 C C 8 - 4 F 6 E - 9 5 7 D - 6 5 C C 6 E 8 D D 6 D 4
�  kfrmID  $ 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H C 8 B 6 1 E F C - D 2 3 D - 4 C A 7 - B 9 4 8 - 5 7 D B F C 9 A 1 6 0 6
�� kfrmID  % 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H D 2 D 8 C 6 2 B - 9 8 F A - 4 8 9 C - B 9 0 C - 9 5 5 4 B 1 6 B 9 B B 4
�� kfrmID  & 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H D 6 0 A 4 A 6 E - 5 7 8 4 - 4 5 B B - 8 A 7 B - C D 7 F A 8 1 2 5 D 2 C
�� kfrmID  ' 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H A 4 C 6 D 8 F 2 - B 5 F 1 - 4 9 4 0 - 9 9 0 E - 4 2 8 E 1 C 8 7 1 3 A 6
�� kfrmID  ( 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H F 5 1 2 D 5 E A - F 0 1 2 - 4 F E C - B 6 E 0 - A 3 D C 5 7 8 2 2 8 F D
�� kfrmID  ) 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 2 9 6 F E 3 9 A - 0 3 5 1 - 4 B A 6 - 9 4 D 8 - 3 C 1 6 4 E C 5 A 3 7 1
�� kfrmID  * 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H D 1 5 9 C 8 4 1 - 2 B 3 4 - 4 A B 3 - B F 0 7 - 2 4 6 3 F 2 3 B 5 1 3 2
�� kfrmID  + 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H B 7 4 6 8 7 8 9 - 7 0 4 0 - 4 4 D 8 - 9 4 A 0 - 0 4 4 3 E B 4 2 1 E 1 8
�� kfrmID  , 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 2 C 6 8 F 5 6 2 - 1 B 7 3 - 4 D 8 5 - 9 1 1 F - 0 3 E D E 8 F 8 0 6 9 1
�� kfrmID  - 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H F 4 9 F 0 9 1 8 - 9 4 D B - 4 8 B 8 - 8 5 4 F - F 3 0 F 3 2 4 3 9 9 D C
�� kfrmID  . 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 9 A 5 4 2 4 5 0 - 5 2 3 7 - 4 5 5 F - A D 9 D - 8 1 A 1 A 8 3 E 1 D E 2
�� kfrmID  / 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H F 6 B 8 C 8 8 B - 4 B E 0 - 4 F C 1 - 8 B E F - C 6 D E D 7 0 6 E D 3 E
�� kfrmID  0 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 4 E 2 1 1 7 C 8 - A 5 E 8 - 4 E 1 0 - A 8 4 3 - E A E C C 9 6 0 7 2 B 9
�� kfrmID  1 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 2 4 8 1 B 6 9 7 - 2 7 C 0 - 4 7 8 C - B D C 1 - 1 9 0 4 0 0 9 5 9 0 D C
�� kfrmID  2 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H C B 0 C 9 0 F 5 - B 2 6 9 - 4 9 1 E - 8 D 6 4 - 3 C 2 9 2 9 2 F B 8 4 C
�� kfrmID  3 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 4 9 3 3 C 3 1 1 - A 7 3 B - 4 5 1 A - B 3 8 9 - B 9 5 1 7 7 7 7 A 3 D 0
�� kfrmID  4 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H F 8 0 A A 8 0 8 - D 8 E 2 - 4 B 7 6 - 8 9 5 7 - 6 A 0 D 4 2 D E E 9 3 0
�� kfrmID  5 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 8 E 1 A 3 1 1 B - B E 4 1 - 4 D C 7 - 8 2 6 5 - 9 3 A 8 9 F 4 D E F 7 0
�� kfrmID  6 	�	� 	���	���	� b��	���
�� 
wres	� �	�	� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev	� �	�	� H 0 C C C 9 0 5 D - 1 B E 7 - 4 D 4 3 - 9 6 9 8 - 6 0 7 9 6 B 4 7 4 E 9 8
�� kfrmID  7 
 
  
��
��
 b��
��
�� 
wres
 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
 �

 H B 4 C B B E 7 5 - 6 D 4 D - 4 E 1 B - B B C 4 - 2 7 6 2 2 F 8 F 8 1 5 F
�� kfrmID  8 

 
��
��
 b��
	��
�� 
wres
	 �



 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
 �

 H 1 8 5 1 9 1 3 E - F 2 1 B - 4 D D D - 9 C D 9 - 9 3 A 4 7 4 1 E 7 F 7 9
�� kfrmID  9 

 
��
��
 b��
��
�� 
wres
 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
 �

 H B 7 9 7 E C 4 A - 0 6 A F - 4 1 4 E - 9 F 6 3 - 0 7 E 5 5 3 4 A 4 5 2 5
�� kfrmID  : 

 
��
��
 b��
��
�� 
wres
 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
 �

 H 0 6 3 B 6 D F 3 - C 8 4 F - 4 E 7 9 - A 6 9 F - 5 C 0 5 4 4 6 D 5 B C 0
�� kfrmID  ; 

 
��
��
 b��
��
�� 
wres
 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
 �

 H B 1 1 4 E B 3 1 - 8 C 6 D - 4 0 4 3 - 8 5 F 5 - F 0 7 C 8 9 8 D 0 0 C E
�� kfrmID  < 

 
��
 ��
 b��
!��
�� 
wres
! �
"
" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
  �
#
# H 8 C 4 D 8 9 D B - 5 3 5 D - 4 3 E 0 - 8 3 4 8 - 9 B A C A 3 8 3 7 6 2 B
�� kfrmID  = 
$
$ 
%��
&��
% b��
'��
�� 
wres
' �
(
( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
& �
)
) H 6 5 D 6 5 C 8 A - 3 0 4 8 - 4 5 3 F - 9 F 2 6 - 6 C 0 5 3 E 8 9 5 5 F 0
�� kfrmID  > 
*
* 
+��
,��
+ b��
-��
�� 
wres
- �
.
. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
, �
/
/ H D 3 1 7 5 9 A E - F 5 3 7 - 4 1 1 8 - 8 F 4 5 - 8 0 8 9 C 7 6 0 3 3 1 B
�� kfrmID  ? 
0
0 
1��
2��
1 b��
3��
�� 
wres
3 �
4
4 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
2 �
5
5 H 1 6 E 0 1 5 6 4 - D E B 6 - 4 1 8 F - 9 D A D - 8 8 3 A 2 9 E D F 5 1 F
�� kfrmID  @ 
6
6 
7��
8��
7 b��
9��
�� 
wres
9 �
:
: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
8 �
;
; H C 6 2 D 6 D A E - 8 B 6 E - 4 2 E 1 - A E 0 4 - 1 6 8 4 B E A 0 C F C D
�� kfrmID  A 
<
< 
=��
>��
= b��
?��
�� 
wres
? �
@
@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
> �
A
A H 3 0 F A 2 3 C B - 0 C B 9 - 4 1 8 9 - A B 2 D - 3 B F 5 3 D A F 7 1 D 6
�� kfrmID  B 
B
B 
C��
D��
C b��
E��
�� 
wres
E �
F
F H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev
D �
G
G H B 6 5 4 2 2 0 6 - D 9 4 E - 4 C 4 5 - 9 1 C 5 - 4 C 0 1 F 8 6 B B 7 A 2
�� kfrmID  C 
H
H 
I��
J��
I b�
K�~
� 
wres
K �
L
L H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrev
J �
M
M H F D 7 0 3 3 8 1 - 0 E D C - 4 6 D D - A B 8 3 - 3 7 A E F E 8 A 6 3 9 6
�� kfrmID  D 
N
N 
O�}
P�|
O b�{
Q�z
�{ 
wres
Q �
R
R H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev
P �
S
S H 1 6 C 5 2 5 F 2 - 8 6 0 3 - 4 B 6 A - 8 8 0 4 - 1 E A D 1 5 F D 9 9 8 6
�| kfrmID  E 
T
T 
U�y
V�x
U b�w
W�v
�w 
wres
W �
X
X H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev
V �
Y
Y H 2 C C A 3 F B 4 - 1 E B D - 4 D B 2 - B 7 8 9 - 8 8 1 3 6 7 A A D B B 5
�x kfrmID  F 
Z
Z 
[�u
\�t
[ b�s
]�r
�s 
wres
] �
^
^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev
\ �
_
_ H A 7 A F F 1 8 C - 9 9 B 1 - 4 1 1 1 - 9 C 4 5 - B E 6 6 6 C C 1 0 0 F D
�t kfrmID  G 
`
` 
a�q
b�p
a b�o
c�n
�o 
wres
c �
d
d H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev
b �
e
e H D C D C 2 B 7 6 - 0 D F E - 4 E E 0 - B 6 0 4 - D D 7 9 D 7 5 F B 2 7 6
�p kfrmID  H 
f
f 
g�m
h�l
g b�k
i�j
�k 
wres
i �
j
j H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev
h �
k
k H 0 A D E 9 9 C 8 - 4 9 1 C - 4 6 E D - A 7 6 C - 3 1 7 5 6 7 F 4 7 A 0 0
�l kfrmID  I 
l
l 
m�i
n�h
m b�g
o�f
�g 
wres
o �
p
p H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev
n �
q
q H 2 6 9 1 E 7 7 2 - F F 3 9 - 4 9 3 B - B 1 7 7 - 2 8 3 4 D 7 C E A 9 3 7
�h kfrmID  J 
r
r 
s�e
t�d
s b�c
u�b
�c 
wres
u �
v
v H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev
t �
w
w H C F 0 C 7 9 F 7 - 9 5 8 5 - 4 3 6 4 - 8 D 1 A - 6 5 0 A 3 8 C B 4 9 1 4
�d kfrmID  K 
x
x 
y�a
z�`
y b�_
{�^
�_ 
wres
{ �
|
| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev
z �
}
} H F A C 3 8 E A 0 - A 7 8 A - 4 C D 5 - 9 0 B E - 6 3 8 D 6 F 8 A 3 D 4 A
�` kfrmID  L 
~
~ 
�]
��\
 b�[
��Z
�[ 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev
� �
�
� H B C 5 D 5 E 7 0 - 3 6 6 3 - 4 8 3 E - A A C 1 - 7 E 7 D 6 6 F 1 5 A A C
�\ kfrmID  M 
�
� 
��Y
��X
� b�W
��V
�W 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev
� �
�
� H D 6 A A 4 D 0 B - F B 7 B - 4 9 0 D - A 8 F C - 5 2 4 B A B 2 0 7 6 0 6
�X kfrmID  N 
�
� 
��U
��T
� b�S
��R
�S 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev
� �
�
� H 5 5 A D B 8 E 6 - 6 C 3 9 - 4 D 0 E - B 3 C 5 - 7 A 1 0 1 E B A 3 7 0 8
�T kfrmID  O 
�
� 
��Q
��P
� b�O
��N
�O 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev
� �
�
� H D 7 1 0 2 D B A - F 0 7 9 - 4 1 5 8 - A 8 5 3 - 9 B A D 0 5 C 7 0 8 F 2
�P kfrmID  P 
�
� 
��M
��L
� b�K
��J
�K 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev
� �
�
� H 8 9 5 2 0 D B 5 - 3 9 F 0 - 4 1 6 0 - 9 9 8 5 - 2 3 8 C 1 0 F B F 9 8 F
�L kfrmID  Q 
�
� 
��I
��H
� b�G
��F
�G 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev
� �
�
� H 3 E 0 9 A 7 3 3 - E 3 2 9 - 4 C 2 A - 8 F 3 1 - B 8 8 5 A 3 C 4 9 7 2 C
�H kfrmID  R 
�
� 
��E
��D
� b�C
��B
�C 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev
� �
�
� H 4 4 C F E E E A - 1 A 0 8 - 4 B 1 8 - B 3 1 2 - 8 2 3 0 6 4 2 B C 8 A F
�D kfrmID  S 
�
� 
��A
��@
� b�?
��>
�? 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev
� �
�
� H 6 5 9 E C 0 4 5 - 7 3 5 9 - 4 F D F - 8 B C F - F 1 3 4 D 9 2 E 2 7 1 D
�@ kfrmID  T 
�
� 
��=
��<
� b�;
��:
�; 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev
� �
�
� H 7 6 1 9 9 0 2 7 - B 4 0 B - 4 1 6 E - 8 E 3 D - 8 F 5 0 1 3 A B C B 0 E
�< kfrmID  U 
�
� 
��9
��8
� b�7
��6
�7 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev
� �
�
� H 9 0 8 1 E 1 E A - A 4 7 8 - 4 4 A 9 - A 5 D 5 - 4 E F 3 1 F 5 9 C C E 1
�8 kfrmID  V 
�
� 
��5
��4
� b�3
��2
�3 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev
� �
�
� H 1 0 B E 0 B 9 D - E F 0 A - 4 D 6 9 - 8 7 4 B - 5 C 4 C D E 9 0 F 9 C 4
�4 kfrmID  W 
�
� 
��1
��0
� b�/
��.
�/ 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev
� �
�
� H A 8 0 F 9 8 9 A - F F 1 6 - 4 1 A D - B 9 D 8 - D 2 D 6 9 0 0 5 E 4 F F
�0 kfrmID  X 
�
� 
��-
��,
� b�+
��*
�+ 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev
� �
�
� H 0 8 9 3 F E E 4 - 5 D C 4 - 4 D 2 A - B 1 9 8 - 1 C 2 5 2 B 3 9 B 5 C 2
�, kfrmID  Y 
�
� 
��)
��(
� b�'
��&
�' 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev
� �
�
� H D C 4 D 8 4 C 2 - E F A D - 4 0 9 A - B 8 2 E - 8 F 3 0 3 7 E D 0 7 8 3
�( kfrmID  Z 
�
� 
��%
��$
� b�#
��"
�# 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev
� �
�
� H 0 8 D C C E 5 A - A F 2 E - 4 3 5 1 - A D D E - 9 F 1 A 6 B E A E 9 E 3
�$ kfrmID  [ 
�
� 
��!
�� 
� b�
��
� 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev
� �
�
� H 6 1 D 8 E 4 F 8 - 2 F 5 4 - 4 E 7 E - 9 C 6 2 - 8 3 D F 3 2 0 A D 1 9 9
�  kfrmID  \ 
�
� 
��
��
� b�
��
� 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev
� �
�
� H 3 5 8 C 9 6 9 D - A 5 4 A - 4 8 7 C - 8 D E C - 4 E E B 4 4 9 7 0 4 2 E
� kfrmID  ] 
�
� 
��
��
� b�
��
� 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev
� �
�
� H 3 C A 3 9 B F F - 3 C D 6 - 4 B 4 3 - 8 2 D 7 - 1 4 5 B 0 1 A 0 1 5 6 D
� kfrmID  ^ 
�
� 
��
��
� b�
��
� 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev
� �
�
� H 6 5 3 7 F D 8 0 - 5 3 C 6 - 4 C 7 4 - 8 A 8 5 - E 2 1 7 4 F 7 1 1 9 E 0
� kfrmID  _ 
�
� 
��
��
� b�
��
� 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev
� �
�
� H 4 0 C 6 8 A F 7 - 1 4 E 5 - 4 E 4 6 - B 5 0 8 - C 2 6 A D 4 0 9 5 E 3 A
� kfrmID  ` 
�
� 
��
��
� b�
��

� 
wres
� �
�
� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev
� �
�
� H 2 8 E 4 0 0 8 6 - A 7 0 5 - 4 5 2 2 - 8 6 2 7 - 9 4 9 5 2 4 1 3 6 5 0 6
� kfrmID  a 
�
� 
��	
��
� b�
��
� 
wres
� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev
� � H 0 4 7 1 D B B C - A 9 5 6 - 4 5 3 B - B D E 8 - 3 E 0 B 8 6 7 B 4 1 0 7
� kfrmID  b  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 8 3 7 1 1 2 9 3 - B C C E - 4 D 5 9 - 9 E E 8 - 1 8 A 5 B D 1 E 8 8 3 9
� kfrmID  c  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H 8 6 8 6 8 A 6 4 - 6 4 7 C - 4 9 1 5 - 8 5 4 5 - 5 5 F 0 F 3 3 9 3 2 3 D
�  kfrmID  d  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  e  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H C 4 4 8 1 1 4 4 - 6 F 1 2 - 4 2 E 3 - 8 8 2 9 - 7 5 D B F 7 5 7 2 0 6 F
�� kfrmID  f  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 2 E 7 0 3 5 8 3 - D 2 C 1 - 4 6 0 7 - B 3 4 C - 8 8 E 9 A 9 C 0 A B 9 2
�� kfrmID  g    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H 2 1 9 B 0 1 C B - 6 6 A 3 - 4 6 E 9 - B C 6 D - 8 4 A 2 A 0 1 4 2 1 E D
�� kfrmID  h && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H 5 E D 1 D 6 8 1 - 1 E 8 6 - 4 C 0 6 - 8 9 8 A - F 5 0 C E 5 2 2 E F 7 4
�� kfrmID  i ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 3 C 6 E 1 A E 9 - D 5 D 8 - 4 2 9 C - B E 0 0 - C 2 6 5 A B 6 E 3 8 C B
�� kfrmID  j 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H 6 8 7 7 8 D 6 A - F 6 4 B - 4 D E C - 9 3 3 9 - 4 3 4 A D 2 9 C 1 C C C
�� kfrmID  k 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H C 1 D 6 C 3 F A - F 5 A C - 4 E 6 C - 9 1 0 F - 5 4 9 2 2 D 3 2 1 C B 6
�� kfrmID  l >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H 2 7 E 4 3 A B F - 5 2 7 0 - 4 4 B F - 9 D 8 F - 8 8 5 C B 4 0 6 0 5 3 B
�� kfrmID  m DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H 4 7 3 7 C D 0 1 - 6 A A F - 4 7 6 C - A 0 C F - F 7 4 2 F 0 6 B 6 3 2 A
�� kfrmID  n JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H 8 D 3 9 0 1 3 3 - 2 2 D 1 - 4 7 E 1 - 8 7 2 7 - 2 7 C 7 6 6 7 C A E 0 5
�� kfrmID  o PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H D B 3 7 2 4 D 3 - F E F 2 - 4 D 2 1 - A F 6 7 - E D 0 1 0 1 8 A 9 E D 4
�� kfrmID  p VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H 5 B B F 3 3 7 D - 2 7 6 B - 4 1 5 7 - 9 6 C F - E E D 3 9 7 3 5 C C B 8
�� kfrmID  q \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H 3 7 D 1 E 9 D 4 - E F 9 3 - 4 9 C A - A 7 F D - D 5 A 0 8 C 0 A 3 1 F 2
�� kfrmID  r bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 0 D 8 8 2 F E 1 - C 1 C 1 - 4 9 D B - 9 9 1 9 - 1 7 D 4 5 8 C 6 0 8 A 9
�� kfrmID  s hh i��j��i b��k��
�� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevj �mm H 2 A 7 E 4 F 7 2 - F D F 9 - 4 4 9 F - B 0 9 4 - E 5 6 1 8 B 3 7 9 3 1 2
�� kfrmID  t nn o��p��o b��q��
�� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevp �ss H 4 7 0 8 6 0 0 9 - 0 3 B B - 4 7 8 D - B B 7 5 - F 1 1 1 2 7 F 0 6 7 5 A
�� kfrmID  u tt u��v��u b��w��
�� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevv �yy H 7 A 0 3 B 4 1 F - 5 D F 5 - 4 C 3 1 - A B 1 A - 6 C 6 B 8 2 7 7 E B F 9
�� kfrmID  v zz {��|��{ b��}��
�� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev| � H 5 1 8 5 7 C 3 F - A B D F - 4 2 5 4 - A D A 2 - B 0 4 E 1 1 7 B 2 D 1 E
�� kfrmID  w �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 2 B 9 0 2 E E - F 0 E 3 - 4 2 1 E - B 9 8 2 - 2 8 1 2 0 0 3 3 A F 5 A
�� kfrmID  x �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 8 1 C 8 2 4 2 - E 1 D 6 - 4 3 D 0 - A C 6 0 - 1 9 6 A 2 1 D 3 6 4 B 3
�� kfrmID  y �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 1 9 6 8 A A B - 6 0 0 3 - 4 E B 7 - A 2 E E - 2 F 4 0 9 3 4 D A 8 7 F
�� kfrmID  z �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 5 C 6 9 2 1 4 - 1 2 6 6 - 4 B 4 4 - B 7 8 9 - 4 B 8 3 8 2 9 0 4 2 F 4
�� kfrmID  { �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 3 B 3 F 3 2 1 - 6 C 0 7 - 4 9 6 5 - 8 3 B C - 0 C 5 7 3 F 3 C 4 2 D B
�� kfrmID  | �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 1 1 5 3 9 F B - 7 B 8 D - 4 6 3 D - A 3 D 9 - F E 8 8 C 2 3 F C F 7 F
�� kfrmID  } �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 4 6 8 8 D 2 C - 2 F 3 9 - 4 E D E - B 6 8 2 - A E 6 D A 1 0 4 4 C 5 7
�� kfrmID  ~ �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 6 8 B 0 9 C 6 - 0 9 4 C - 4 8 E D - B E 9 7 - 5 A 5 4 E 3 8 B C 7 0 4
�� kfrmID   �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 A 9 9 7 4 0 4 - 2 B A D - 4 A 0 B - A 8 E 4 - D 8 C 5 8 A 2 D A 4 5 9
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A A 6 8 8 E C 8 - 2 5 4 E - 4 6 2 6 - 8 C 7 5 - 6 E 7 5 5 F 8 D 7 A F 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 3 0 B 5 1 F 2 - 5 6 7 8 - 4 6 A B - 9 9 6 9 - 1 1 0 D 8 2 8 9 B 7 4 C
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D E 4 5 4 1 8 D - D 4 2 5 - 4 2 8 2 - A 5 9 6 - 6 3 A 2 2 7 6 A 3 4 6 B
�� kfrmID  � �� ������� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrev� ��� H B D 4 A 8 4 A 0 - 1 C B 8 - 4 A D 9 - A B 3 5 - 7 1 8 C 9 7 7 8 1 7 F D
�� kfrmID  � �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H C E 9 C F 8 4 6 - 4 9 A 6 - 4 C E 3 - A 7 0 C - 0 1 3 E 7 E B E 8 F 3 B
�| kfrmID  � �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H C 7 6 6 A 5 A A - 3 6 7 C - 4 7 1 8 - 9 D B 1 - D D D E B 5 C 5 D 5 5 3
�x kfrmID  � �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H 3 B 4 D 9 0 6 F - 1 4 A 9 - 4 9 A 1 - 9 E B 5 - D 7 3 7 C 1 5 E 4 F 2 9
�t kfrmID  � �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H F B 2 A 0 F 3 7 - A 4 4 3 - 4 B B 4 - A D E 5 - F C 2 C C C 3 7 F 3 D 8
�p kfrmID  � �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 7 7 B 3 D 0 B 6 - 1 C B B - 4 7 4 0 - 9 3 8 4 - 3 F C E F F A C 3 0 9 4
�l kfrmID  � �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H B A 9 A 4 D B 2 - 8 3 F 6 - 4 9 D 3 - 8 1 3 D - 1 0 E 2 6 B 5 C 9 D 1 F
�h kfrmID  � �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H F 5 5 6 9 4 1 A - 2 E F 4 - 4 6 9 1 - A 2 2 2 - 2 6 9 B 3 D A 3 5 E 4 8
�d kfrmID  � �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H A 9 0 5 3 E B 4 - 0 9 B 0 - 4 7 F 1 - B 0 F 2 - 6 9 E F 7 9 9 5 B B 6 2
�` kfrmID  � �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H 9 0 8 1 E 1 E A - A 4 7 8 - 4 4 A 9 - A 5 D 5 - 4 E F 3 1 F 5 9 C C E 1
�\ kfrmID  �  �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H 7 A 0 3 B 4 1 F - 5 D F 5 - 4 C 3 1 - A B 1 A - 6 C 6 B 8 2 7 7 E B F 9
�X kfrmID  � 

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H 7 A 0 3 B 4 1 F - 5 D F 5 - 4 C 3 1 - A B 1 A - 6 C 6 B 8 2 7 7 E B F 9
�T kfrmID  �  �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H 7 A 0 3 B 4 1 F - 5 D F 5 - 4 C 3 1 - A B 1 A - 6 C 6 B 8 2 7 7 E B F 9
�P kfrmID  �  �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 5 E D 1 D 6 8 1 - 1 E 8 6 - 4 C 0 6 - 8 9 8 A - F 5 0 C E 5 2 2 E F 7 4
�L kfrmID  �  �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H 5 E D 1 D 6 8 1 - 1 E 8 6 - 4 C 0 6 - 8 9 8 A - F 5 0 C E 5 2 2 E F 7 4
�H kfrmID  � "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�D kfrmID  � (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�@ kfrmID  � .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�< kfrmID  � 44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�8 kfrmID  � :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�4 kfrmID  � @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�0 kfrmID  � FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�, kfrmID  � LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�( kfrmID  � RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�$ kfrmID  � XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H C 3 B 3 F 3 2 1 - 6 C 0 7 - 4 9 6 5 - 8 3 B C - 0 C 5 7 3 F 3 C 4 2 D B
�  kfrmID  � ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H 1 0 B E 0 B 9 D - E F 0 A - 4 D 6 9 - 8 7 4 B - 5 C 4 C D E 9 0 F 9 C 4
� kfrmID  � dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H 6 2 B 9 0 2 E E - F 0 E 3 - 4 2 1 E - B 9 8 2 - 2 8 1 2 0 0 3 3 A F 5 A
� kfrmID  � jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H 5 1 8 5 7 C 3 F - A B D F - 4 2 5 4 - A D A 2 - B 0 4 E 1 1 7 B 2 D 1 E
� kfrmID  � pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
� kfrmID  � vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
� kfrmID  � || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
� kfrmID  � �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�  kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 3 B 3 F 3 2 1 - 6 C 0 7 - 4 9 6 5 - 8 3 B C - 0 C 5 7 3 F 3 C 4 2 D B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 C C A 3 F B 4 - 1 E B D - 4 D B 2 - B 7 8 9 - 8 8 1 3 6 7 A A D B B 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 7 A F F 1 8 C - 9 9 B 1 - 4 1 1 1 - 9 C 4 5 - B E 6 6 6 C C 1 0 0 F D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 7 A F F 1 8 C - 9 9 B 1 - 4 1 1 1 - 9 C 4 5 - B E 6 6 6 C C 1 0 0 F D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 9 9 2 1 0 4 F - C 6 F A - 4 E 2 B - 8 D F 6 - E 8 E 0 3 9 7 6 D 8 9 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 7 0 8 0 1 4 C - E 3 2 8 - 4 E D 0 - 9 F 9 2 - 9 A E 2 6 8 A 4 6 C A 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 9 5 4 6 6 7 5 - 5 1 B 3 - 4 7 6 0 - 9 B 8 1 - 6 E C 1 7 0 E 0 8 8 7 A
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 4 7 4 0 0 E 3 - C 3 E F - 4 0 3 D - 8 3 7 0 - B B 5 A 1 4 1 6 7 D 2 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 8 6 4 F 6 1 E - 9 7 E 0 - 4 C 9 C - B 3 4 1 - F 3 4 6 1 9 5 C B B A 3
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 6 8 A A 7 5 E - C F E 4 - 4 A D 1 - 9 4 8 3 - E 2 4 0 4 0 2 7 5 4 3 D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 2 7 F 1 F B 4 - 6 1 C F - 4 F C 8 - A 4 7 C - 6 3 C 2 6 A D 5 F 1 1 F
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 9 B D 8 E F 3 - 7 E C 2 - 4 1 D 5 - 8 F 2 4 - B 0 6 8 1 5 8 1 8 E A 7
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 A 0 3 B 4 1 F - 5 D F 5 - 4 C 3 1 - A B 1 A - 6 C 6 B 8 2 7 7 E B F 9
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 7 A F F 1 8 C - 9 9 B 1 - 4 1 1 1 - 9 C 4 5 - B E 6 6 6 C C 1 0 0 F D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 A 6 0 C 1 A 5 - F 7 C C - 4 1 D 8 - B C 0 F - 5 2 3 B B 3 9 B 3 0 9 6
�� kfrmID  �    ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H B F 0 1 A 3 A F - 1 3 C 2 - 4 2 7 3 - 8 E E 1 - 7 2 6 8 C 2 B A D 3 5 D
�� kfrmID  �  ���� b��	��
�� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H F A D 3 2 C 9 B - 0 7 0 E - 4 D 3 5 - 8 3 E 4 - 3 4 8 6 8 5 3 9 3 2 C 5
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H A E E 1 1 F 8 8 - 2 4 8 E - 4 F A D - B C 0 0 - 6 D C 0 1 7 A 9 4 7 F E
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 7 5 7 B 3 0 7 3 - D 1 3 6 - 4 2 D 2 - 9 9 B 6 - B E F B C 3 5 A 3 4 6 F
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 8 A 7 4 0 D 9 F - 7 9 D A - 4 7 1 D - A A 1 B - D 3 1 C 6 C 9 E 0 F 8 6
�� kfrmID  �  �� �� b��!��
�� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �## H 0 1 6 C 9 9 4 4 - 0 7 7 D - 4 0 A 1 - 9 1 8 8 - 6 F 6 F 7 E D 3 C 5 5 9
�� kfrmID  � $$ %��&��% b��'��
�� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �)) H 6 D B 4 C F 3 B - E 1 F E - 4 1 E C - B 5 1 7 - 8 B E D E A 8 A D 1 6 B
�� kfrmID  � ** +��,��+ b��-��
�� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev, �// H F 2 A 6 6 7 8 A - 8 6 C 5 - 4 0 6 3 - 8 B C A - 3 E 0 9 C 0 E 3 B 2 4 7
�� kfrmID  � 00 1��2��1 b��3��
�� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev2 �55 H A C C 8 7 6 7 1 - 0 1 3 A - 4 E D 4 - 9 4 6 9 - F 1 8 A 0 7 E 6 E B 0 7
�� kfrmID  � 66 7��8��7 b��9��
�� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev8 �;; H D 5 F D C 6 9 6 - B D 7 5 - 4 B D 6 - 9 F E 5 - 2 1 1 8 4 2 3 1 F E 8 A
�� kfrmID  � << =��>��= b��?��
�� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev> �AA H 2 B E B 5 B 8 4 - 2 B C 9 - 4 1 8 0 - B 1 1 7 - 7 D 7 A B 5 7 C E 3 5 9
�� kfrmID  � BB C��D��C b��E��
�� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevD �GG H C 1 D 1 E 8 6 B - 9 1 D 4 - 4 F 2 4 - 9 4 7 F - 1 0 9 4 D E C C 2 5 F B
�� kfrmID  � HH I��J��I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrevJ �MM H B 3 1 A 2 A C B - 9 C C D - 4 2 D B - A 0 A 7 - 1 6 E E D B 2 0 D 7 1 D
�� kfrmID  � NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H 9 7 1 3 F E D 6 - 5 B 1 D - 4 6 4 C - A F 4 3 - 7 F 2 F 4 F 4 3 5 6 1 8
�| kfrmID  � TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H D 2 C 6 A 2 5 B - 6 4 B 1 - 4 2 1 0 - B 6 A 6 - 0 B C 1 8 F 0 B 1 7 4 B
�x kfrmID  � ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H A 8 0 7 A 7 A 2 - 5 D E 7 - 4 F 5 2 - B D 7 F - B E 6 E 9 9 2 E 9 6 5 B
�t kfrmID  � `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H 1 E 3 6 1 A 2 8 - 3 E 4 6 - 4 D 5 C - A F C F - 2 F 3 9 7 7 8 8 1 5 A 8
�p kfrmID  � ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H 2 5 F 5 8 5 4 9 - 6 1 8 0 - 4 4 2 7 - A 0 7 D - 7 1 6 A 9 6 F E 0 C A A
�l kfrmID  � ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H F 7 4 D 7 4 6 3 - 5 2 A 5 - 4 7 6 9 - A 9 3 A - E 1 A 6 C 9 8 5 6 0 D 6
�h kfrmID  � rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H F 6 A 8 C B 5 3 - E 0 0 8 - 4 B C 6 - B 7 5 7 - C 4 6 F B 9 D 2 1 3 6 2
�d kfrmID  � xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H 2 2 B 0 5 E 7 2 - 8 A 0 8 - 4 5 5 8 - B 4 D 4 - 4 6 8 3 A 3 0 5 F 4 7 B
�` kfrmID  � ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 7 C F 7 C B 5 9 - 8 1 C 7 - 4 3 E B - 9 D B F - F 2 0 E 9 1 B A 2 9 8 9
�\ kfrmID  � �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H 9 7 5 8 8 E D 2 - E 4 F A - 4 1 2 4 - 8 0 8 2 - 4 A 9 D 8 2 4 8 6 7 C 3
�X kfrmID  � �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H 9 3 E 3 6 9 9 0 - 5 A 2 D - 4 6 5 0 - 8 5 E D - 0 A E B 8 D 1 B E 6 7 F
�T kfrmID  � �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H 6 2 E 8 9 3 E 1 - E 8 9 1 - 4 5 2 8 - B 4 0 E - 3 8 A 5 3 8 9 3 A B 2 0
�P kfrmID  � �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H A 0 7 D 1 F 2 5 - 9 7 F 7 - 4 B C D - 8 A F 6 - 9 9 5 B 8 6 8 F 2 1 0 9
�L kfrmID  � �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H 3 5 1 7 2 A 0 C - D 5 2 6 - 4 F E B - 8 F 3 7 - 8 D B 0 B 8 8 9 C 2 2 E
�H kfrmID  � �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H D 3 3 6 7 D 7 D - D 1 8 6 - 4 B 4 D - 9 4 7 8 - 4 5 F 2 4 2 8 4 A 8 D 8
�D kfrmID  � �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�@ kfrmID  � �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H 5 2 E 9 2 8 8 C - 6 8 D 6 - 4 F 2 F - 8 D 5 5 - 0 5 7 3 B F 7 A E 5 2 C
�< kfrmID  � �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H B 0 B B 0 7 C 3 - 4 7 9 9 - 4 8 D 9 - B 1 4 F - 0 8 2 9 1 D 0 9 9 8 4 E
�8 kfrmID  � �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H 9 D 1 6 4 E A C - 2 5 0 E - 4 A 4 3 - B 2 8 0 - 7 7 A 6 2 E B C D A 1 0
�4 kfrmID  � �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H 7 2 8 3 E 5 C B - 0 3 6 9 - 4 B C 6 - 9 A 1 F - 8 5 D 0 A 8 1 D C 8 3 C
�0 kfrmID  � �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H 2 D B 3 9 E A 3 - 6 A B 6 - 4 3 9 0 - 8 9 5 9 - F A 4 9 2 D A 4 D B 9 9
�, kfrmID  � �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H C D 0 1 9 7 C 4 - 8 9 E 5 - 4 8 D 9 - B 6 5 A - 9 0 A 7 1 1 8 A 1 D A 0
�( kfrmID  � �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H 5 3 2 3 F 0 2 4 - 2 6 9 5 - 4 C B 7 - B 7 2 8 - B 3 B 6 F A E 9 0 8 1 3
�$ kfrmID  � �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H F 2 1 0 E 0 5 C - 3 E 1 D - 4 8 9 4 - 9 0 4 8 - 5 1 5 8 5 8 1 5 C 8 F E
�  kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 3 4 0 3 E B C - 9 B F E - 4 C 0 A - B A A E - 5 A B 1 6 0 B 1 A F B 5
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E B 2 5 7 1 7 5 - 4 2 B F - 4 C 3 F - 8 8 D 5 - 9 E B 9 F 6 1 7 1 D 7 5
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E 9 1 3 2 4 1 7 - F A 9 8 - 4 7 1 C - 8 C E C - B 9 2 9 4 5 3 8 3 6 1 9
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 0 2 E 2 8 8 9 - D F 7 6 - 4 B 5 6 - A 5 6 7 - 4 3 4 3 5 C 9 0 2 E B 1
� kfrmID  � �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H 7 7 D 2 A 5 6 1 - E 3 A 7 - 4 9 9 E - B 9 A 3 - 3 B 8 E E 0 5 C 7 C A 3
� kfrmID  � �� ��	��� b���
� 
wres� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� � H B D B E B 3 B 2 - D 6 0 F - 4 8 6 5 - 8 5 F B - 7 9 5 9 F E 8 6 F B 2 1
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H C 9 9 7 9 C 7 4 - 1 A C A - 4 E 5 C - B C F 9 - 7 1 B 5 C 7 B 3 9 4 E 2
� kfrmID  �  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H F B 9 6 D C F 5 - F 9 E 6 - 4 E A 2 - 8 3 8 C - E 9 6 9 0 9 F 1 C 8 9 D
�  kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H E D D 8 4 5 E 1 - 5 E 1 1 - 4 6 8 9 - A F D 8 - C C 9 3 2 9 C 3 3 3 1 0
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 6 C 7 8 F 1 6 2 - 0 4 2 D - 4 7 6 A - A D A 0 - 0 8 4 1 2 A D 0 E B 3 0
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 8 6 9 8 3 3 B 1 - 4 A 0 1 - 4 B F C - B 7 3 7 - 3 1 8 0 0 C D 0 1 0 B D
�� kfrmID  �    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H 5 C 8 2 1 C 0 C - 9 0 1 8 - 4 2 C B - B C 7 1 - E 3 F A E A 2 B C 4 D 7
�� kfrmID  � && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H 6 6 0 1 A 7 4 6 - D E F 9 - 4 6 5 1 - 8 2 2 7 - 2 D C 5 9 D 3 8 1 1 E 0
�� kfrmID  � ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 4 0 C F A A B E - B 5 8 4 - 4 F B C - 8 1 D 4 - F D 9 1 4 8 C E 9 F C F
�� kfrmID  � 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H 7 9 F C 8 E 3 6 - C E D F - 4 9 9 1 - A 9 8 F - 8 A E E 8 F E 6 8 D E A
�� kfrmID  � 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H 0 5 E 4 9 2 6 6 - 5 F B A - 4 7 5 5 - B 1 0 8 - 6 D 0 F 8 8 7 F 8 4 7 C
�� kfrmID  � >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H 3 D B C C D B 5 - 9 E 3 0 - 4 8 9 E - B 4 0 6 - 6 6 B 7 4 D 1 0 8 1 2 A
�� kfrmID  � DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H C 2 6 D 2 9 A 4 - 8 F 5 7 - 4 0 1 F - A 8 F D - 6 2 1 F 8 7 D 0 C C 1 D
�� kfrmID  � JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H F 8 7 1 4 F 7 F - 8 8 3 D - 4 A 6 6 - A 6 D 4 - 8 E E 0 B 5 B A B C A 1
�� kfrmID  � PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H 7 5 5 E 1 6 E 6 - 1 9 1 D - 4 9 C B - B 2 2 A - A 3 4 C 0 6 2 D B C A B
�� kfrmID  � VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H 5 6 C 2 1 6 1 4 - 7 7 7 F - 4 8 A 9 - 8 0 D 9 - D A E E A F 0 E 1 9 7 3
�� kfrmID  � \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H A 6 D C 1 2 3 F - D E 3 C - 4 2 1 2 - B 2 0 4 - D E 8 E F 2 7 1 5 5 0 D
�� kfrmID  � bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 5 9 1 0 8 1 6 6 - 8 3 2 9 - 4 D F B - A C 7 4 - D F F 2 0 5 3 8 7 8 6 9
�� kfrmID  � hh i��j��i b��k��
�� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevj �mm H 5 F F A 3 9 C B - D 7 E 9 - 4 F A B - A 2 0 E - D 3 D 4 6 8 2 4 4 6 7 4
�� kfrmID  � nn o��p��o b��q��
�� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevp �ss H F A C 4 4 4 7 5 - A 7 E B - 4 C B 5 - A D C C - 2 4 C F F A 3 2 D 7 0 8
�� kfrmID  � tt u��v��u b��w��
�� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevv �yy H F 1 D D E 5 9 5 - 7 9 3 A - 4 A 2 5 - B 9 0 A - 4 D 5 F E 0 2 1 A A 6 C
�� kfrmID  � zz {��|��{ b��}��
�� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev| � H 2 F 4 2 8 F 3 2 - E 5 E C - 4 B 6 8 - A 0 2 0 - 9 1 4 F A 3 9 A C 7 9 B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 3 9 C B 6 E 0 - B 2 5 D - 4 E A A - 8 4 4 0 - 1 6 0 6 E 0 F A D F A B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 F A F C 9 3 7 - 1 0 6 C - 4 A C 3 - B 7 B 9 - 9 C 1 5 8 1 9 E 2 6 A 4
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 B 0 8 5 3 9 5 - 2 5 9 9 - 4 B E C - 8 5 0 8 - A 4 A 4 8 1 D D 4 C 0 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 3 0 C A 1 A 6 9 - 1 8 4 2 - 4 0 1 D - 9 2 A 8 - 6 D 0 B 2 D 0 4 5 C 3 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E B C B 8 8 9 E - 6 5 C F - 4 8 C C - A E 0 3 - 4 0 E 6 B 8 B A F 5 B F
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C C 1 C B 6 C 2 - 5 8 3 E - 4 C D D - 8 0 4 B - 9 E 1 1 C E D C A A 7 7
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F 9 B 9 4 4 E 3 - 5 F B 3 - 4 F 3 F - B 4 B 8 - A E A D 2 C E A D D 3 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E 9 6 6 8 1 4 A - 3 B 2 B - 4 5 3 7 - B 4 6 4 - 3 3 5 F 9 2 4 F 5 9 9 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 3 0 6 7 2 A E - F 8 7 0 - 4 1 B 5 - B B B 3 - 9 9 4 D D 7 0 5 5 B C F
�� kfrmID    �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 2 6 7 9 5 1 1 - 9 2 F 2 - 4 1 C 1 - 8 6 2 B - 0 3 C 5 6 B 4 6 8 E 8 D
�� kfrmID   �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 B 9 F 5 5 E 6 - 8 5 8 F - 4 D E F - 8 C 2 C - 3 1 E 9 4 7 8 0 C A D 2
�� kfrmID   �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A E 5 1 4 F 4 C - B 5 E 2 - 4 D 8 5 - 9 C B 5 - 1 A E 6 F F B D 0 4 9 0
�� kfrmID   �� ������� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrev� ��� H E D 7 7 E E 6 7 - 3 9 8 F - 4 A 0 5 - 8 8 A D - 1 2 F A 4 B C 0 0 A C 1
�� kfrmID   �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H 3 1 B F 5 4 C C - A 5 7 B - 4 9 2 3 - 9 E 7 4 - 8 C 9 F 9 5 B 8 D E 2 1
�| kfrmID   �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H 9 0 5 B 0 A 9 8 - 5 7 A E - 4 4 8 B - B 3 8 9 - C C 7 1 E 7 B B 5 2 6 C
�x kfrmID   �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H 1 A 9 1 D 1 C E - C 3 2 C - 4 5 8 3 - A 8 B 0 - 1 4 5 C 0 3 F C B E F 8
�t kfrmID   �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H E 6 A D 6 E C B - D 2 B 4 - 4 6 B F - B E 4 4 - 5 F F 6 5 E 4 E 7 E 9 D
�p kfrmID   �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 2 2 D 9 9 5 6 0 - B 0 B 1 - 4 A A 6 - A 5 5 7 - B 3 9 4 5 B 5 B 1 7 8 A
�l kfrmID  	 �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H 1 B 0 5 2 F 4 2 - C 3 3 8 - 4 A B 7 - B 4 6 C - 9 2 4 F 2 1 5 E 2 6 9 1
�h kfrmID  
 �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H 7 3 0 8 A 7 8 1 - 1 2 B 2 - 4 4 C 5 - A 1 1 C - E 3 F 9 D 1 2 F 0 3 8 A
�d kfrmID   �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H D 4 0 8 C A 9 B - F 2 B E - 4 7 4 9 - B 3 2 E - B C 5 2 9 8 E 9 8 9 7 B
�` kfrmID   �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H D E 3 D A 4 C 6 - E E A 6 - 4 4 E 3 - 8 4 0 9 - 5 4 5 3 1 A C 8 F A 5 E
�\ kfrmID    �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H D 5 9 6 6 8 0 D - 5 E F B - 4 D A E - A A F F - 1 5 1 F F 8 C E C 4 F 5
�X kfrmID   

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H 7 8 C C 9 6 7 0 - A A 9 7 - 4 F 9 1 - 8 6 5 8 - E 9 5 C 7 6 8 8 0 4 5 1
�T kfrmID    �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H 0 9 E 8 4 8 F 5 - F 0 0 E - 4 9 E 2 - B 7 3 0 - 8 D 1 F D E 4 3 C B C 6
�P kfrmID    �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 1 4 F 0 6 8 8 8 - 5 0 E E - 4 1 3 C - 9 D 8 B - 7 D C 9 E A B 8 2 1 1 D
�L kfrmID    �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H 8 5 B 1 B F 2 6 - E 7 D 7 - 4 D 0 8 - 9 A 6 C - 1 8 D 4 A 1 8 4 9 4 6 3
�H kfrmID   "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H 7 4 4 0 B B B A - 8 3 4 C - 4 C 5 3 - 8 D E 8 - 6 E 6 0 5 7 3 F 0 3 1 D
�D kfrmID   (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H 5 9 8 E 3 7 6 D - B C C 2 - 4 5 A 7 - A 8 B A - D 2 0 C 9 0 9 D C 5 9 C
�@ kfrmID   .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H B 0 C 4 4 7 D 4 - 2 E 4 C - 4 F D 0 - 9 7 5 0 - D 7 C 0 9 4 B 9 9 2 9 A
�< kfrmID   44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H D 9 A D E D 9 1 - 3 E 2 8 - 4 8 A 7 - 9 E 9 D - C D 6 2 1 5 D C 6 B A 7
�8 kfrmID   :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H A 3 1 D 6 2 1 4 - 4 B 1 5 - 4 4 9 5 - 8 9 B F - 8 E 7 4 4 A D 8 9 8 9 0
�4 kfrmID   @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H E 5 1 4 6 8 7 8 - C C 0 1 - 4 7 5 8 - 9 1 5 2 - D B 3 9 8 4 2 4 6 3 B 7
�0 kfrmID   FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H 9 D D 8 C A 1 5 - 5 E B 7 - 4 0 7 1 - B F 3 2 - D 7 E 5 5 A B 4 4 5 F E
�, kfrmID   LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H D 4 8 9 6 8 8 8 - 2 8 4 4 - 4 A 0 F - A E 8 9 - 9 E A 0 1 7 F 5 7 A 9 7
�( kfrmID   RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H 4 E 1 6 D 7 D 0 - D 4 D A - 4 E 1 C - B C 0 A - D B C 1 6 A 5 B 8 E F 9
�$ kfrmID   XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H C F 6 6 0 0 1 4 - 8 3 4 3 - 4 7 0 9 - B 1 F 7 - 7 2 0 7 4 1 6 6 B 0 0 5
�  kfrmID   ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H 5 0 8 F 8 C D 6 - 3 0 2 9 - 4 B 1 F - B 4 1 A - 6 1 4 5 1 9 A A 3 1 A 9
� kfrmID   dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H 1 5 E 2 5 4 4 9 - F 1 9 5 - 4 D B C - B 1 C 5 - 3 1 0 4 6 7 F 1 5 3 7 2
� kfrmID   jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H B 8 8 1 4 2 0 F - 7 7 6 0 - 4 6 5 8 - B 4 D 6 - F 1 6 2 2 D 4 B 8 D 9 C
� kfrmID   pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H E 5 8 5 F 4 9 E - 4 4 0 B - 4 C 6 3 - 9 6 1 E - 5 E 0 B 8 0 5 8 D 2 0 B
� kfrmID    vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H F 5 B B E 5 1 4 - E 2 B 9 - 4 9 2 B - A E 6 C - 4 5 3 9 A 0 4 9 E D D A
� kfrmID  ! || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H D A 3 3 2 D 7 C - 6 5 C 2 - 4 E A 0 - 8 F A 7 - 2 8 2 B F 1 8 5 F 0 C 8
� kfrmID  " �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D F E 5 0 F E 9 - E 2 E 2 - 4 6 B 0 - 9 2 4 B - 3 6 4 5 7 1 F 4 4 7 0 C
� kfrmID  # �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H 5 4 E 3 D 5 C E - 9 2 C A - 4 4 3 E - B 1 F 4 - B D 6 8 2 E 1 1 4 1 3 C
�  kfrmID  $ �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 4 2 5 3 8 0 B - 7 8 E C - 4 5 4 9 - A 2 6 E - C 5 5 A 0 0 5 A 6 1 D 7
�� kfrmID  % �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D B 0 3 8 5 4 4 - 1 4 3 A - 4 A 5 3 - A 2 8 2 - F 9 E C 3 4 B 6 F 7 5 8
�� kfrmID  & �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B A 1 B 6 9 E C - A D 3 A - 4 D 8 8 - 9 6 D 0 - C F D 3 6 E 6 7 8 5 0 3
�� kfrmID  ' �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 4 D A E 5 D 6 - F C B B - 4 4 E 0 - 9 0 8 C - 2 A C 4 1 D 2 E A 2 2 8
�� kfrmID  ( �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 5 8 6 9 9 F 0 - B F 8 A - 4 5 B 7 - 9 D D 6 - F 3 C A 3 5 7 C D 5 1 7
�� kfrmID  ) �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 2 E 8 9 3 E 1 - E 8 9 1 - 4 5 2 8 - B 4 0 E - 3 8 A 5 3 8 9 3 A B 2 0
�� kfrmID  * �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F 1 6 E A 3 1 A - C C 3 E - 4 A B 3 - 9 D A 6 - 3 0 6 7 E 0 D 7 9 3 9 2
�� kfrmID  + �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 8 D E F F F 2 - 5 1 1 B - 4 F A 1 - B 7 0 7 - C F A 4 5 7 5 1 B C 7 5
�� kfrmID  , �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 9 8 1 B F 2 2 - 3 E 5 B - 4 5 E F - 8 A 3 B - 9 8 5 7 E 2 F 8 5 8 0 B
�� kfrmID  - �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F A 2 8 0 9 2 D - 2 2 8 B - 4 6 1 B - 9 0 C 5 - F 6 1 C D D 7 A C A 7 6
�� kfrmID  . �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 7 5 F E 1 9 7 - 8 E C A - 4 1 0 B - 8 E E A - 4 3 4 D 1 F 3 2 E C 0 6
�� kfrmID  / �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 1 1 4 D B 0 3 - 8 6 1 B - 4 7 3 F - B A 4 0 - F 5 A B 2 2 9 8 6 5 8 D
�� kfrmID  0 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 8 4 6 E 7 E 8 - F 6 F 2 - 4 8 2 1 - B 1 9 B - 7 0 9 0 3 9 7 8 0 E 3 1
�� kfrmID  1 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E 9 1 9 1 4 6 6 - 6 A 9 C - 4 F 1 C - 8 B 7 4 - F A A 1 E D 0 5 A 5 9 B
�� kfrmID  2 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 B C E A 9 9 2 - C 2 4 F - 4 0 C E - A F 6 1 - 0 9 0 F B 1 5 D 8 B 8 3
�� kfrmID  3 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 C D D 2 9 7 0 - B 1 E 8 - 4 F 0 E - 9 B 7 D - 1 1 D 1 F 1 B E 2 D 2 7
�� kfrmID  4 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 E 3 6 6 1 B 9 - B 9 4 3 - 4 8 8 6 - A 4 3 5 - 0 C 0 D 7 D 3 0 A D 4 B
�� kfrmID  5 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 5 0 5 E 4 F C - F A 2 9 - 4 2 5 6 - B 6 3 6 - 7 E 9 F 4 D B D 1 9 5 A
�� kfrmID  6 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 2 A F 3 5 3 3 - D A F 8 - 4 7 3 7 - 8 F D 9 - C A 0 C D C 8 5 E 5 A A
�� kfrmID  7    ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H D E 9 F 9 4 7 A - 1 7 4 9 - 4 4 4 2 - 8 2 0 8 - A 6 E 7 4 B 3 3 E F 9 4
�� kfrmID  8  ���� b��	��
�� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H E F A 2 2 8 4 1 - 4 1 3 4 - 4 B 1 1 - A 6 C 6 - 9 0 8 8 1 5 8 A 9 E 2 0
�� kfrmID  9  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 3 1 6 B 6 5 F 6 - 9 8 F 2 - 4 9 9 7 - 8 A F E - 2 4 B E 1 B 1 4 4 6 B E
�� kfrmID  :  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 7 3 C 2 E B F 3 - 7 8 5 A - 4 B 9 9 - 9 E C 2 - 5 C 0 C 4 4 9 E F B 5 0
�� kfrmID  ;  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 9 6 1 9 3 B F 4 - 4 8 C 3 - 4 8 3 1 - A 5 B 6 - 6 2 2 F 4 8 C 1 D 0 E 2
�� kfrmID  <  �� �� b��!��
�� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �## H 9 F 2 B 5 B 7 0 - 2 E 5 7 - 4 A D 1 - B F E D - B A 0 5 0 A 8 8 4 9 9 3
�� kfrmID  = $$ %��&��% b��'��
�� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �)) H 8 4 6 B 1 6 A 6 - 6 7 B 5 - 4 7 1 9 - A 6 E B - C B D 9 3 C 1 2 E 0 7 4
�� kfrmID  > ** +��,��+ b��-��
�� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev, �// H 8 4 1 7 8 3 3 9 - 5 A 8 7 - 4 2 A A - 8 A 1 F - 0 2 C 3 A 5 2 0 D C 3 A
�� kfrmID  ? 00 1��2��1 b��3��
�� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev2 �55 H D C B F 7 D 5 8 - C 7 3 A - 4 8 E F - 8 2 3 6 - 2 C 4 8 5 C 8 5 2 8 C 2
�� kfrmID  @ 66 7��8��7 b��9��
�� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev8 �;; H 1 A 8 1 2 4 3 E - 1 E F E - 4 E 3 3 - 8 6 6 B - 9 F 9 2 8 F 7 3 7 1 C 6
�� kfrmID  A << =��>��= b��?��
�� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev> �AA H 2 6 9 A F D C 7 - D 9 A 7 - 4 D 0 C - B 2 F E - 4 7 C D 6 6 2 5 8 E 5 E
�� kfrmID  B BB C��D��C b��E��
�� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevD �GG H 9 7 5 8 8 E D 2 - E 4 F A - 4 1 2 4 - 8 0 8 2 - 4 A 9 D 8 2 4 8 6 7 C 3
�� kfrmID  C HH I��J��I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrevJ �MM H 3 5 9 9 0 F D C - E F B F - 4 3 E B - 8 3 0 0 - F 9 8 D 0 C 5 3 E 3 7 5
�� kfrmID  D NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H 6 1 0 7 0 2 A B - A A 6 1 - 4 5 B 5 - B 4 D A - 4 C 4 B B 2 4 D 9 5 4 A
�| kfrmID  E TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H F A 4 8 1 2 E 1 - 0 5 8 F - 4 A 1 8 - 9 6 D 9 - 8 8 2 1 6 9 7 C B 6 0 1
�x kfrmID  F ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H 7 6 3 C D E 4 4 - E 6 4 E - 4 B B 3 - B 5 9 0 - C 5 4 4 9 E F 3 0 6 F 8
�t kfrmID  G `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H 4 C 3 8 C C D 0 - 0 5 5 C - 4 D 9 6 - B F C 1 - 0 B 9 A 2 0 E 2 F 9 9 9
�p kfrmID  H ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H A 8 0 A 1 A 4 A - 9 B 3 3 - 4 F 4 7 - 9 2 1 8 - A 2 E 1 0 F 3 A F C 9 7
�l kfrmID  I ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H 5 D D 8 C 3 A 8 - C F 2 5 - 4 5 B 1 - B C E 0 - 7 E C A 9 E 6 4 1 B 9 F
�h kfrmID  J rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H E 6 7 1 8 F F 6 - 8 2 F 3 - 4 0 5 7 - 8 9 6 9 - E D 2 D E 6 B 3 C 6 8 5
�d kfrmID  K xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H 3 B D 0 1 9 3 E - C 3 4 E - 4 F 8 A - 9 D 6 F - 5 A 9 A 5 C 2 C D 2 1 8
�` kfrmID  L ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 3 5 F C F 5 7 A - 5 5 C 9 - 4 A F 9 - 9 D 0 F - 1 1 C 5 A 6 3 C 4 9 9 1
�\ kfrmID  M �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H 0 C C E 4 5 D 2 - 7 2 D 8 - 4 4 C 0 - A 8 2 D - 4 3 E 5 3 5 1 F F D 3 2
�X kfrmID  N �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H 0 0 C 0 F B F D - F F 6 E - 4 A 9 5 - A 2 1 B - 7 2 D 4 9 3 3 5 6 F C E
�T kfrmID  O �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H 9 8 8 9 D 7 6 6 - 0 0 6 7 - 4 9 7 5 - 8 3 C 6 - E 3 D 6 9 1 6 F F 4 9 B
�P kfrmID  P �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H 8 A 8 2 B 7 A D - B E 0 0 - 4 2 E 2 - 9 7 D 7 - D 0 C 2 4 3 C A B 9 0 D
�L kfrmID  Q �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H 1 B 8 2 E 5 7 D - 1 B A C - 4 5 1 D - 9 0 1 6 - 4 0 9 8 2 0 E 5 D 1 5 8
�H kfrmID  R �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H 2 3 C E F 7 3 1 - F B A 1 - 4 A 7 8 - 9 3 B C - 1 4 9 E 0 1 0 1 9 A C 8
�D kfrmID  S �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H 3 5 0 5 E 5 A 4 - 0 6 0 8 - 4 5 2 9 - B D 9 9 - 0 9 4 6 2 B 0 5 E 1 8 0
�@ kfrmID  T �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H F 2 5 3 4 5 A 8 - E E 1 6 - 4 E A B - 8 A 2 9 - 0 4 E E 6 5 0 3 B F A 9
�< kfrmID  U �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H 8 9 A 8 5 5 D 9 - C 2 A D - 4 1 A 0 - A 1 5 9 - 1 1 1 0 3 B 5 D 0 6 9 9
�8 kfrmID  V �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H F 0 2 E E 8 A 3 - 7 1 4 F - 4 5 2 D - 8 F 7 2 - 0 1 9 8 A C 7 A C 5 C A
�4 kfrmID  W �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H 7 9 3 7 7 7 1 A - D 3 F 8 - 4 6 C 3 - A B F 2 - 8 B C C 9 2 E 1 9 6 8 9
�0 kfrmID  X �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H 8 1 5 0 C F 0 F - 7 A E 9 - 4 7 B 2 - B 8 7 D - B B 6 7 1 E 6 1 9 6 E 6
�, kfrmID  Y �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H 6 7 B 1 7 6 7 4 - 8 0 2 E - 4 8 1 1 - A 0 F 7 - 2 9 6 0 5 3 B 9 C C E F
�( kfrmID  Z �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H 2 1 6 9 D 2 C D - 5 5 8 6 - 4 3 3 3 - 9 9 B 8 - 5 D A A F 4 3 3 9 B A 1
�$ kfrmID  [ �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H D 6 2 A B E 2 1 - C C 8 E - 4 C B B - A B 4 C - 3 9 F 6 0 2 B E E D 7 3
�  kfrmID  \ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 4 D D 0 2 4 5 - 7 6 0 F - 4 0 B 7 - 8 7 9 D - 7 1 F 3 9 1 A 2 0 F 7 2
� kfrmID  ] �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 F 2 5 F A 7 B - E E F 3 - 4 9 D 4 - 9 5 6 4 - 4 9 A 7 F F 9 5 2 F 7 8
� kfrmID  ^ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 8 2 1 9 4 4 8 - 1 1 D 6 - 4 B B A - B 0 E C - F E E 7 9 4 9 B B 0 C 6
� kfrmID  _ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E 2 C C 2 1 C 4 - 8 5 A 0 - 4 F A 0 - B 6 B 5 - 1 A F B B 6 0 5 1 C 5 5
� kfrmID  ` �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H A 1 9 5 4 3 9 A - 8 0 5 4 - 4 D 6 A - 9 5 7 8 - 5 A 2 E 6 8 2 9 9 4 8 9
� kfrmID  a �� ��	��� b���
� 
wres� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� � H 2 C 3 8 F F 6 6 - 3 F A 9 - 4 8 0 3 - A D 5 6 - B 2 3 D 1 9 D 2 B 7 4 0
� kfrmID  b  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H C A 9 9 C B 3 9 - B D 7 B - 4 3 2 9 - B F 4 B - 7 6 D 2 6 B 9 4 5 C B B
� kfrmID  c  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H C 2 9 E 0 7 7 9 - 7 D A D - 4 8 0 2 - A 2 E C - 0 9 3 F A 7 D 8 D D 2 3
�  kfrmID  d  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H E B 3 2 3 6 6 3 - 9 7 9 8 - 4 3 5 E - 8 B 7 4 - 6 4 C 6 2 4 2 6 C C C C
�� kfrmID  e  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H D E C 1 3 F 5 3 - 6 3 1 4 - 4 1 F B - A 0 C 7 - 1 6 0 5 6 3 A 0 2 6 3 B
�� kfrmID  f  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H D 5 E 0 3 F A 5 - F 8 8 9 - 4 B F D - A 2 E C - B 4 8 F 2 B 0 A F 0 E B
�� kfrmID  g    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H 5 8 F B B 4 9 4 - 6 6 6 2 - 4 8 D E - 8 D 3 1 - 1 F 0 6 1 4 5 B F 9 B 8
�� kfrmID  h && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H A E 5 2 2 6 6 E - F 9 8 6 - 4 A C 0 - 8 F D C - 0 6 8 7 7 D F B 5 E E 9
�� kfrmID  i ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 4 6 9 8 7 F 2 6 - D E 4 F - 4 D B C - 8 D D 8 - A 2 F 9 9 7 3 6 2 B 7 B
�� kfrmID  j 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H 5 6 4 D 0 4 E F - 7 9 2 0 - 4 8 6 9 - 9 6 E 5 - 8 4 A F 8 C B F B 7 B 2
�� kfrmID  k 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H 4 D C 7 9 D 7 A - 4 4 A 5 - 4 6 E D - A 9 C E - 8 1 7 5 F E 8 3 7 D 0 B
�� kfrmID  l >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H 9 0 F A 3 F 6 6 - 5 3 4 8 - 4 9 8 0 - B D 5 3 - 4 9 C 8 B 2 C B 4 6 F B
�� kfrmID  m DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H 3 F D 9 A F 7 4 - C 1 9 9 - 4 F 2 D - B 0 8 2 - C 2 9 F 1 5 3 3 6 6 E 2
�� kfrmID  n JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H 2 1 3 A 5 C E 9 - F D 1 E - 4 0 9 D - 9 8 4 A - 0 5 9 E 3 F D B C 7 1 C
�� kfrmID  o PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H E 7 C 4 A C D F - 4 D 4 F - 4 E 8 0 - 9 3 F 1 - 6 9 6 8 B B D 3 E 4 3 8
�� kfrmID  p VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H 3 5 8 4 A D C 1 - 6 F 1 B - 4 4 1 F - B 3 0 D - A F C 3 1 9 4 3 2 7 1 0
�� kfrmID  q \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H B 7 5 C 2 1 7 A - 4 E 0 A - 4 D 3 4 - 8 7 C 6 - F 8 1 B 0 5 D 3 D 3 0 3
�� kfrmID  r bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 5 6 4 C 4 B 2 B - 2 7 1 B - 4 6 0 A - 9 F B 2 - E 8 9 D D 4 6 0 3 A 8 F
�� kfrmID  s hh i��j��i b��k��
�� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevj �mm H 9 4 C E 1 D 6 2 - 5 E A 8 - 4 0 9 A - 8 0 E A - 4 C 7 C C 2 C 6 7 E 1 E
�� kfrmID  t nn o��p��o b��q��
�� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevp �ss H C 2 2 9 6 E 5 7 - 9 F E C - 4 7 A 1 - 8 E 8 7 - 8 A A 2 2 5 2 9 6 F 1 B
�� kfrmID  u tt u��v��u b��w��
�� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevv �yy H F C 1 2 4 4 C F - 6 C F 6 - 4 D 5 C - 8 F 1 5 - 0 7 2 E 8 5 5 C 4 5 E 6
�� kfrmID  v zz {��|��{ b��}��
�� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev| � H 4 7 2 2 5 B 1 9 - D 9 2 D - 4 5 4 2 - 9 B E 0 - 8 5 C 5 3 A 0 8 6 3 B B
�� kfrmID  w �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 5 2 F 5 B 6 1 - 8 8 B E - 4 E D E - A F C E - A C 6 4 8 0 8 4 1 5 F 3
�� kfrmID  x �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 1 3 D F 9 5 5 - 8 4 6 8 - 4 1 A E - 9 6 9 D - 2 B 8 4 7 0 3 3 0 B F 5
�� kfrmID  y �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 5 D 3 6 3 B D - 1 F 0 1 - 4 A 9 4 - B 7 5 C - 7 B 5 C 8 3 B 3 A 5 1 B
�� kfrmID  z �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 6 8 C A 8 C 9 - 0 5 A 3 - 4 3 F C - 8 A 4 5 - 7 3 F 7 6 8 C 6 F A C 0
�� kfrmID  { �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 D 0 1 0 8 E F - 0 2 1 F - 4 A A 6 - B 9 6 1 - 9 6 2 1 6 7 0 A 6 0 3 5
�� kfrmID  | �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 7 7 F 9 7 6 3 - A 7 C F - 4 C A 8 - A 3 0 6 - D 0 2 F 1 7 E 4 6 9 0 1
�� kfrmID  } �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 5 F 7 F A 6 2 - F F F E - 4 9 4 B - B 2 1 5 - 7 2 0 8 0 F 5 D 7 D 9 8
�� kfrmID  ~ �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 D B 9 9 0 1 E - 8 0 7 2 - 4 4 5 F - 9 A 3 4 - 0 8 9 5 4 E 7 C 6 7 7 C
�� kfrmID   �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 3 0 9 4 3 7 A - 5 5 1 2 - 4 B E C - 9 0 5 A - 6 C 4 5 9 9 A 6 8 9 7 E
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 8 4 1 C E 0 2 - 6 7 4 7 - 4 F D E - A 6 5 9 - 3 9 9 2 2 1 D 2 4 1 A 3
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 F D B A A E 3 - 3 B E 6 - 4 B F F - 9 5 3 3 - 1 D D 1 2 1 E C 9 2 2 C
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A B 3 3 1 A 8 3 - 2 1 C C - 4 7 9 0 - B E 6 5 - 6 2 F 3 C D 9 2 0 0 0 1
�� kfrmID  � �� ������� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrev� ��� H E 9 3 0 3 6 F F - 7 B C 2 - 4 8 0 E - 9 F 6 3 - D 0 2 4 E 4 4 B B 5 0 3
�� kfrmID  � �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H 2 C B C 9 4 A 5 - 8 7 2 F - 4 0 D F - 9 9 A 1 - C E F 4 2 0 4 7 3 2 5 9
�| kfrmID  � �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H 4 C 0 6 7 7 0 A - 8 B E 3 - 4 E 8 1 - 9 7 0 8 - F 4 1 8 5 A 4 3 A 8 3 4
�x kfrmID  � �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H 2 4 2 C A D 7 9 - 1 0 2 D - 4 6 F D - A 7 1 F - 4 0 E 9 F 2 F 8 C 7 1 0
�t kfrmID  � �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H F 3 8 E 7 C 6 9 - 3 D F B - 4 3 5 6 - A 0 1 3 - 2 3 C D 9 E 0 A C D 0 3
�p kfrmID  � �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 9 7 0 4 B 3 C 1 - 0 C A 2 - 4 1 7 A - 9 9 A F - 8 4 9 E B 5 E 6 B 0 A 0
�l kfrmID  � �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H 5 7 1 7 5 1 4 D - 9 9 7 2 - 4 5 F 8 - 9 A F D - 6 2 7 2 3 5 5 9 5 3 D 2
�h kfrmID  � �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H A 5 E 2 7 0 4 3 - 4 6 7 0 - 4 8 F B - 8 0 A E - 4 5 F 1 3 B B A 2 2 7 3
�d kfrmID  � �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H E 2 0 6 E 0 4 F - 9 3 F D - 4 1 F D - 8 B 4 9 - 1 F E F 4 7 7 3 E 8 C 6
�` kfrmID  � �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H B B D 5 4 F 4 C - 1 E 1 B - 4 7 E E - 8 5 D F - 4 E B 5 6 F A 6 3 8 A 5
�\ kfrmID  �  �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H B 8 B 2 7 B 6 4 - 4 6 B C - 4 C 2 2 - B 2 5 C - 2 B 9 9 A 3 7 6 B 7 C 7
�X kfrmID  � 

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H 3 B 6 6 4 D 0 F - 0 D F C - 4 7 4 2 - 8 6 7 F - 3 E 7 6 A 9 2 A 0 4 1 E
�T kfrmID  �  �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H 3 8 8 3 0 5 A C - 6 B 6 8 - 4 C 2 3 - 8 C 5 1 - E F B 0 7 7 F 8 D E 7 2
�P kfrmID  �  �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 9 5 F 4 B E 4 A - 7 A 4 A - 4 2 C 1 - B B 6 F - 0 8 2 D D 3 C 3 9 9 B B
�L kfrmID  �  �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H C 6 E D C E D 8 - B C D C - 4 7 2 F - 9 1 E B - 5 8 C C 6 E 2 5 2 C F 6
�H kfrmID  � "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H A 9 9 A 4 A F 6 - 8 4 5 9 - 4 3 6 F - 9 A 1 A - 7 F 2 3 5 8 B A 5 6 E 1
�D kfrmID  � (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H 6 6 7 3 A F 8 6 - D 0 6 1 - 4 A F B - 8 7 4 E - 5 3 C B C 1 7 0 7 7 3 B
�@ kfrmID  � .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H A A E 1 D 2 A 2 - C D A 4 - 4 6 F 9 - 9 9 F 2 - 6 B E E A 5 C A 9 B C 3
�< kfrmID  � 44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H 8 5 C 2 F 9 2 A - D 5 4 B - 4 D 9 2 - B B E 3 - E D 6 4 3 E 5 C F 5 7 0
�8 kfrmID  � :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H 9 7 8 F 5 B A B - 1 2 3 D - 4 B A C - 8 4 0 7 - 3 4 4 D 2 2 8 0 6 A 5 D
�4 kfrmID  � @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H 5 8 5 F 8 9 C 7 - 1 F C 3 - 4 0 A 7 - 9 C B 4 - D 7 8 5 F 0 2 4 B F 9 A
�0 kfrmID  � FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H 2 3 3 2 0 F C 2 - 6 2 2 2 - 4 6 5 D - 8 C 5 E - 4 7 7 E E 7 3 E 3 5 4 E
�, kfrmID  � LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H 0 6 6 6 9 4 8 0 - A 9 9 A - 4 6 8 C - B 6 C 2 - 4 9 4 2 5 B 6 0 B 9 9 F
�( kfrmID  � RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H 5 D 1 4 1 D 6 F - B A 5 6 - 4 2 2 C - B F 4 C - 3 9 6 C 7 A 0 B A 2 B 5
�$ kfrmID  � XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H 0 6 D A A C 9 6 - D C 9 2 - 4 4 6 D - 8 B C D - 6 5 8 B 0 3 5 B 5 3 B 0
�  kfrmID  � ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H A 4 4 4 0 7 E D - 0 6 5 7 - 4 F 9 9 - 8 E 3 1 - F B 6 A 3 E 7 E C B 3 D
� kfrmID  � dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H 5 A 2 2 5 7 3 C - 2 6 0 5 - 4 4 2 2 - 9 A A 2 - 4 C B D 5 B 4 0 B E 2 4
� kfrmID  � jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H E B 5 D 6 C 1 2 - 5 7 F 3 - 4 8 4 5 - 9 E 3 A - 8 0 7 5 9 F F 8 2 E 2 9
� kfrmID  � pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H E 5 A F 9 D F 1 - 7 7 4 F - 4 3 8 F - 8 9 C 4 - 9 4 E 3 D 3 A F 1 D 9 8
� kfrmID  � vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H 9 E A C 4 F D 4 - B 2 5 0 - 4 B A B - B E 2 7 - C 3 E 4 E 7 3 F 9 7 B F
� kfrmID  � || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H 6 D E 0 5 B 6 1 - F 6 7 E - 4 C 9 B - 8 5 F 3 - 8 F 8 F 1 5 A F B 7 5 D
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 1 C B C A 6 F 3 - D 5 3 0 - 4 0 6 0 - B 7 5 6 - C 3 B D B E 4 1 2 F 0 A
� kfrmID  � �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H 4 1 1 2 8 5 2 3 - 4 0 F C - 4 C 5 D - 9 6 9 A - 2 D C 8 0 9 9 B 6 3 E 3
�  kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F C 3 5 3 2 C 2 - 9 6 8 9 - 4 F 6 3 - B 2 E 6 - B D 2 5 1 E 3 7 5 6 F 9
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 B E 2 1 9 A 0 - 7 C 5 F - 4 7 5 4 - 8 7 7 4 - 4 D 0 1 7 4 6 6 C C 7 2
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 8 B 5 D 1 2 1 - D 3 1 3 - 4 3 E 5 - 8 7 F 8 - A D E 9 5 4 7 D 1 8 5 0
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 E 5 E E 6 E 8 - F 9 4 3 - 4 2 A A - A 2 D F - 6 8 2 8 2 4 2 C 3 8 1 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 B A 5 8 B 5 D - 7 3 A 6 - 4 F 7 7 - 8 8 6 0 - 4 7 6 6 B F B A 1 2 E B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 4 F 5 D A E 4 - 0 9 E 1 - 4 3 4 9 - 8 0 E 8 - 5 F D E B 6 5 B 5 7 2 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 3 7 E 8 7 0 6 F - 4 9 7 F - 4 B 8 5 - B A 8 2 - 3 3 9 A 6 4 9 C D A 1 A
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 6 F 0 8 B 1 0 - D 6 C E - 4 6 7 5 - A E A 7 - B D A 2 A 2 7 4 4 C 8 A
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F F B 9 9 9 2 7 - 4 6 0 B - 4 2 9 F - 9 2 2 3 - C 7 F B 8 5 2 D 1 2 3 A
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E 9 1 7 F 2 9 7 - 2 0 A 2 - 4 A 8 2 - B 8 A 1 - A 7 F C 8 F 6 3 6 0 1 1
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F 8 C 3 F 7 4 A - D 4 C B - 4 0 2 7 - B 5 A 6 - 4 5 6 1 9 E C 0 7 7 C D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B C 5 E 6 E 1 B - 7 6 2 C - 4 A 7 4 - B 3 F 2 - F 7 4 B 8 7 D 7 A B 3 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 5 C 2 2 D D 0 - 3 9 6 C - 4 B B 9 - 9 D 6 0 - 7 C 6 A 9 F C 1 3 8 4 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F 3 B F 7 9 4 C - 2 C 6 D - 4 7 C 2 - B B 1 6 - E A 0 0 3 2 E 2 5 E F 2
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 0 E 4 7 4 D E - D 8 5 A - 4 C C 5 - B C 0 3 - 5 1 C 3 3 C 4 A 4 9 9 D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 9 D 0 2 5 4 D 8 - 1 7 A 6 - 4 1 6 2 - B 7 6 A - F 7 F D 7 3 F 1 A 5 C C
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 8 3 6 D 8 0 F - A 7 1 A - 4 2 9 8 - B E 8 B - D 6 0 E C 2 3 8 0 A 2 E
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 4 0 F 2 6 E 8 - 9 2 C C - 4 C 8 3 - 8 C 3 3 - 6 C 4 E 9 A 9 B 8 4 4 B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 0 A 7 7 9 3 4 - 7 1 D 5 - 4 6 B 4 - B 3 A 7 - B F A 1 3 A 0 C 5 C F E
�� kfrmID  �    ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 7 8 5 E 5 9 9 0 - 9 6 7 D - 4 0 D 0 - B F 1 2 - A 9 1 4 D 9 E A C 7 1 2
�� kfrmID  �  ���� b��	��
�� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H E 6 B 2 1 7 6 2 - 5 C C B - 4 D D 3 - B B 6 9 - 0 0 9 C 3 A 6 9 E E 5 C
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 3 6 A 2 1 E 4 1 - 1 0 6 B - 4 4 9 D - 9 3 3 F - 9 8 F 6 2 4 0 5 5 F 9 9
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 3 3 A 1 1 A 8 D - 2 5 2 D - 4 7 1 A - 9 E 8 7 - F 7 3 E 9 B E 0 0 C 3 9
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 6 E C 5 F 4 3 4 - 4 4 6 9 - 4 2 E 7 - 8 1 6 1 - A E C 7 D D 2 1 9 9 E A
�� kfrmID  �  �� �� b��!��
�� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �## H 9 E 3 C 6 9 9 2 - A A E A - 4 F A 7 - A 8 D B - A 3 C 9 5 B 6 0 2 E 6 3
�� kfrmID  � $$ %��&��% b��'��
�� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �)) H D A 0 5 E 4 5 E - C 6 6 5 - 4 0 E 9 - A 3 C A - F 2 C A B E 6 5 2 7 5 A
�� kfrmID  � ** +��,��+ b��-��
�� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev, �// H 6 A C 2 2 C 2 6 - 2 8 9 E - 4 6 B 4 - A D 9 1 - 1 1 1 8 1 5 C 0 3 A 6 9
�� kfrmID  � 00 1��2��1 b��3��
�� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev2 �55 H C 0 9 E 7 5 5 4 - 1 4 3 2 - 4 D A 1 - B 0 B 8 - A E E 5 5 3 C D 4 C A 3
�� kfrmID  � 66 7��8��7 b��9��
�� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev8 �;; H 2 C 3 C B E 0 4 - 8 C 7 B - 4 9 1 8 - A 6 2 1 - 1 A 3 9 C 8 0 1 C 8 C 7
�� kfrmID  � << =��>��= b��?��
�� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev> �AA H 7 2 3 F 7 E A C - 6 E F E - 4 2 A 9 - 8 3 3 4 - B 2 4 2 8 3 D 2 4 7 1 B
�� kfrmID  � BB C��D��C b��E��
�� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevD �GG H 2 6 1 F A 8 C F - 1 C B E - 4 8 2 7 - A C 8 9 - B 3 B B 5 D 3 E C B 4 0
�� kfrmID  � HH I��J��I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrevJ �MM H 8 A 4 5 D 2 4 E - D 4 5 3 - 4 4 B F - 8 A C 0 - 6 6 D 3 7 9 1 A 5 F C 0
�� kfrmID  � NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H 8 6 5 3 7 2 1 E - 4 A 5 3 - 4 6 2 5 - 9 0 E E - 9 A B E 3 8 D 0 F 9 B 0
�| kfrmID  � TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H 9 7 5 8 8 E D 2 - E 4 F A - 4 1 2 4 - 8 0 8 2 - 4 A 9 D 8 2 4 8 6 7 C 3
�x kfrmID  � ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H 3 E 0 B D D 0 A - 6 9 0 4 - 4 B 0 C - A 2 1 9 - 7 3 2 9 4 4 9 6 B D 5 5
�t kfrmID  � `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H 1 B 6 F 0 1 7 3 - F C 2 C - 4 C 7 1 - A 9 6 1 - 1 5 8 A 7 3 0 8 4 7 4 E
�p kfrmID  � ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H C 6 5 3 6 9 2 B - F A 7 6 - 4 8 D 4 - 9 F E F - 3 8 E 6 6 3 8 1 A F 9 2
�l kfrmID  � ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H 4 4 B 6 8 5 E 1 - 4 1 6 4 - 4 2 9 4 - A C D C - 2 A 3 4 E D E 7 0 7 B B
�h kfrmID  � rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H 1 F 0 A F A 4 4 - A 7 8 D - 4 E 8 C - B 4 1 4 - 4 C 3 2 8 E 8 7 4 0 E 4
�d kfrmID  � xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H 2 1 1 9 E E 1 B - 3 C 0 F - 4 C 6 3 - A D 9 0 - 8 9 5 F B E 9 5 6 3 6 E
�` kfrmID  � ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 2 B C 2 7 5 7 B - 4 9 0 D - 4 9 C 0 - 8 A 9 5 - 7 C 4 1 3 3 6 C 8 7 8 D
�\ kfrmID  � �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H 6 E 8 6 8 9 9 9 - 3 5 3 0 - 4 A 6 A - 8 1 8 1 - 4 D 1 0 5 2 B 9 D 4 D E
�X kfrmID  � �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H D B 7 A B 8 6 A - 4 E 5 C - 4 9 F 3 - 9 2 E F - D 8 7 3 F 5 B 9 0 5 E 5
�T kfrmID  � �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H B 7 5 0 4 B F 3 - D 7 8 7 - 4 4 0 8 - B 3 7 9 - D 5 1 2 A 8 A C 6 2 F F
�P kfrmID  � �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H 3 C E 2 C F 0 D - 2 D E 3 - 4 3 7 6 - 8 6 A C - 0 4 9 E 7 A 0 4 4 9 5 3
�L kfrmID  � �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H F D 9 D 8 1 E 0 - 2 5 2 1 - 4 2 B B - 9 7 6 E - 7 E 2 0 4 1 B 8 5 3 A A
�H kfrmID  � �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H 9 0 D 4 E C 5 F - 7 F F 1 - 4 D 4 9 - 9 F 5 E - A C 5 E 1 E D 0 3 6 6 6
�D kfrmID  � �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H 8 2 F A C C C F - D 3 B E - 4 5 7 C - 8 6 5 2 - 3 B 4 9 0 2 C 8 4 9 1 5
�@ kfrmID  � �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H B F C 9 4 F 0 3 - E C 4 0 - 4 B 3 F - 9 1 D 9 - 1 B E 5 1 8 0 F 9 E F 3
�< kfrmID  � �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H 3 9 3 A 7 5 D 6 - B B 2 D - 4 4 8 7 - A 4 1 D - B 4 9 A E 7 D 4 6 A F 5
�8 kfrmID  � �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H 4 9 0 E E B 1 2 - 5 B B E - 4 7 E 3 - 8 5 E 3 - A 6 D F 9 7 E B 6 E 8 1
�4 kfrmID  � �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H 0 B F E 4 2 8 2 - B 1 0 0 - 4 1 9 6 - 8 A D E - 1 B 6 1 C A C 0 F 3 A A
�0 kfrmID  � �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H D 0 0 C 2 1 3 2 - 4 6 7 C - 4 3 2 0 - 8 E D A - 7 6 9 5 6 6 1 F C F F B
�, kfrmID  � �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H D F D E 2 B F 0 - 1 C C F - 4 E E 3 - B 4 9 B - E 2 8 9 6 0 5 8 E B B 9
�( kfrmID  � �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H 2 1 2 9 7 2 8 6 - B 2 5 C - 4 1 B 0 - B F C A - 2 5 5 B 8 4 4 7 8 8 D F
�$ kfrmID  � �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H C 1 B 1 6 3 F 8 - B 8 D 6 - 4 A 6 5 - 9 5 C 2 - F 3 3 1 D 3 C 2 4 E D C
�  kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 3 3 4 F 8 E E 4 - 4 1 B 1 - 4 F 3 B - 9 9 1 4 - 8 3 A C 7 1 4 A 2 F 2 B
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H B 7 9 4 9 6 8 4 - 6 0 3 D - 4 4 4 D - A D E 8 - C 5 E 3 D B 1 F 4 B 9 6
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H F 4 E 3 0 B 2 F - 8 D 9 B - 4 5 4 0 - 8 6 1 9 - 5 E D D 4 1 6 D A F 4 7
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E D 6 7 5 5 B D - B A C E - 4 9 0 7 - B A 9 D - F 6 C 2 F 2 1 7 4 7 7 A
� kfrmID  � �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H 3 5 4 B 0 3 F F - 5 1 9 E - 4 E C 8 - B 9 4 4 - 7 3 6 D 1 3 5 B B 1 2 E
� kfrmID  � �� ��	��� b���
� 
wres� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� � H A 0 6 2 D 6 D B - A E 4 E - 4 6 8 E - B 9 C 0 - B 8 E 5 D 5 4 5 6 2 E 7
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 6 2 6 2 2 9 C 8 - 7 A 5 6 - 4 C 1 F - 8 F 8 5 - 2 8 A D 8 C E 7 D 7 2 6
� kfrmID  �  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H 8 7 D D 5 6 F C - 7 F C 9 - 4 B 6 E - A 3 C 7 - E 3 A F D 1 F 3 1 5 9 2
�  kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H B 7 C 6 3 1 5 6 - 0 B E 5 - 4 5 E 4 - B F B E - C 9 E 4 1 3 4 0 3 2 0 D
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 7 5 5 A F 5 5 9 - 8 A 1 4 - 4 B A E - 9 C E E - D C 2 4 1 9 1 E C 3 A 4
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H D 8 9 8 0 1 B 1 - 2 6 9 5 - 4 2 C F - 9 F B 2 - 8 2 5 F F 6 F 4 3 8 7 B
�� kfrmID  �    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H 9 0 E 6 C 3 2 F - B 3 C 3 - 4 5 E E - B E 5 1 - C 0 E 1 C F C 3 7 9 0 7
�� kfrmID  � && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H F 5 B 5 5 6 6 F - 2 F B 3 - 4 4 9 5 - 8 1 6 0 - 1 B A 7 4 E B 6 4 D B B
�� kfrmID  � ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 2 A 8 3 A 4 B 8 - 6 A 2 E - 4 D 1 6 - A 7 7 8 - 0 7 6 C 4 8 2 7 7 A B 8
�� kfrmID  � 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H B D 5 5 A 1 3 F - C D 4 5 - 4 F F E - B E 7 A - 8 4 C E C F 5 3 4 5 A 1
�� kfrmID  � 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H F B 8 E 7 1 D E - 2 E E A - 4 9 3 F - B 5 6 5 - B E C E 8 1 A D 8 A 2 E
�� kfrmID  � >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H 4 F 4 4 0 9 4 7 - 3 6 4 0 - 4 1 8 D - A C 8 E - A 8 8 C 1 A C 0 5 A A 5
�� kfrmID  � DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H A 6 7 9 1 1 8 3 - 7 1 5 A - 4 0 9 1 - B 9 5 5 - 8 D 7 6 1 D 9 4 1 E 2 3
�� kfrmID  � JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H A 4 A C 5 9 2 D - A 9 2 B - 4 4 D D - A 3 6 0 - 2 1 8 1 B 5 C 0 7 8 C 6
�� kfrmID  � PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H 8 9 4 7 A 0 7 A - C 4 F E - 4 6 8 6 - B E 6 3 - C 9 D F A D B A 4 1 D 5
�� kfrmID  � VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H 9 5 6 F F E F B - C F C 0 - 4 0 2 E - 8 B B 0 - E 7 2 0 9 7 C 8 9 2 E B
�� kfrmID  � \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H F 3 9 F B 1 D 6 - E D 0 F - 4 4 5 D - 9 7 3 9 - E 1 7 1 A 1 D 6 4 6 C 6
�� kfrmID  � bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 1 3 A B F F F 4 - 4 B 3 4 - 4 D 4 B - 9 2 9 6 - E 0 2 7 B 3 7 5 3 9 0 B
�� kfrmID  � hh i��j��i b��k��
�� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevj �mm H 8 4 4 A 0 1 C 8 - E 3 E 1 - 4 1 4 D - 8 1 C A - 4 1 E 0 C 1 5 3 1 E 3 4
�� kfrmID  � nn o��p��o b��q��
�� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevp �ss H 0 6 1 B 7 6 2 C - 4 5 7 9 - 4 8 7 2 - 9 0 2 3 - 9 7 0 6 B C F E A A 9 E
�� kfrmID  � tt u��v��u b��w��
�� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevv �yy H 3 3 C 9 5 5 E 3 - A 3 2 C - 4 C C D - A A D 3 - 0 0 1 B C 6 A 8 3 E E A
�� kfrmID  � zz {��|��{ b��}��
�� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev| � H 1 7 2 D E B C 3 - 1 0 E E - 4 8 8 5 - A A F D - 9 F 8 6 2 4 B C 1 0 E 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 D A 7 E F 2 9 - 9 D 5 7 - 4 2 F A - 9 4 1 F - 1 F 2 9 A 2 5 E 9 1 E 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 0 7 E E 3 F 1 - 1 9 6 7 - 4 3 7 7 - 8 2 B 8 - C 2 5 7 3 A 8 0 F 8 4 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E D B 8 A E 6 6 - F C D 3 - 4 8 8 7 - A A 7 A - 7 0 B F B 0 F 2 6 3 8 4
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 D 6 F 9 5 3 7 - C 0 2 1 - 4 C 2 A - 8 1 E 3 - 3 9 F A A 2 C 2 7 B 6 E
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 B A 8 D 8 3 D - F B 3 5 - 4 4 0 8 - B 0 1 5 - 4 8 B 9 2 3 B 2 0 5 5 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 A 2 C 6 6 A F - E 1 A 1 - 4 9 3 1 - A E D 7 - 1 0 6 E 0 6 8 2 C 5 C B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 6 D 4 6 D 3 E - D 9 F 2 - 4 3 E 8 - A 6 9 2 - B 3 0 E E D 3 6 E 5 4 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 8 4 D C B 1 4 - 3 9 9 6 - 4 4 8 C - A 2 4 2 - A D F 8 0 2 A 6 2 9 4 E
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 A E 2 9 8 F D - 1 E 6 7 - 4 2 F 2 - A 2 B 7 - 9 A 4 7 8 2 8 0 4 1 4 1
�� kfrmID    �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 F 4 C 1 2 E 5 - 1 C 8 C - 4 1 A 6 - 9 6 F 1 - C 8 6 8 B 8 F 4 D 5 F E
�� kfrmID   �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 8 4 7 9 7 7 E - 3 B 5 6 - 4 4 9 8 - A A 1 3 - 9 3 F 3 C D C 3 8 7 1 9
�� kfrmID   �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 A 4 6 1 7 E 3 - 9 F 6 6 - 4 9 F B - B 1 0 F - 2 C 0 4 6 9 A D D E F 1
�� kfrmID   �� ������� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrev� ��� H 2 C E 8 B C 9 D - F D 3 7 - 4 7 0 9 - 9 1 E B - 5 B 1 F E 5 9 F 4 8 8 2
�� kfrmID   �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H 6 5 C F 2 0 D A - D 0 9 E - 4 E 1 1 - A D 3 5 - 1 6 4 F 9 8 D C 1 F 1 0
�| kfrmID   �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H C 1 0 D 0 2 D A - 8 2 D 2 - 4 7 7 7 - A 9 4 B - C E F 9 1 2 1 3 3 9 5 C
�x kfrmID   �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H E 4 6 A E F B 0 - 8 F E 2 - 4 D A 6 - 8 1 9 6 - 4 A C D 0 2 5 9 1 9 D F
�t kfrmID   �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H B 9 2 C A A E 9 - 9 4 E F - 4 8 A 8 - 9 6 6 C - F 2 8 9 6 1 F 8 5 D 9 7
�p kfrmID   �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 4 D 1 C 0 8 F F - 1 6 3 1 - 4 D 1 B - 8 3 9 6 - B 1 A 4 7 D E 6 5 0 0 A
�l kfrmID  	 �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H 3 B 0 A C 5 4 E - A 8 E 0 - 4 A F 0 - 9 D 6 2 - 3 A B C 7 3 1 7 F F 9 7
�h kfrmID  
 �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H 5 C 9 4 B D C 7 - 1 0 3 2 - 4 7 7 9 - 8 7 2 F - B 2 C C A F F 7 9 8 6 8
�d kfrmID   �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H F C D 8 5 B 6 1 - 5 F C E - 4 1 6 3 - A A F 6 - F 6 A D 8 C 3 D 6 1 5 2
�` kfrmID   �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H 1 B D C C 6 5 F - B A D 1 - 4 7 C 2 - B 6 E A - E F 8 E A 3 F E 4 4 A A
�\ kfrmID    �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H 4 A 9 D 2 8 4 1 - F D C 0 - 4 6 0 C - A 9 A 0 - 8 8 F E 3 8 F C 4 5 D A
�X kfrmID   

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H 7 E 7 E A 6 6 5 - 9 2 6 9 - 4 D B 7 - 9 5 4 6 - 4 7 4 A 1 F 9 2 5 1 D 0
�T kfrmID    �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H 4 C F 5 6 E 6 7 - E B 4 9 - 4 C D 8 - B E 0 7 - 2 8 A C 3 C B 1 9 9 6 E
�P kfrmID    �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 5 B C 6 0 5 D F - F 7 8 B - 4 2 1 C - B B 3 0 - D 5 8 E C 8 1 E 0 F D 3
�L kfrmID    �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H 8 4 0 5 9 8 4 0 - D 2 4 D - 4 E E 2 - 9 9 2 B - 1 6 8 4 A A 5 9 D 4 2 8
�H kfrmID   "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H 2 D D 3 E 4 E 5 - 6 E 6 0 - 4 2 E B - B 1 E 5 - 9 2 1 C D F 8 D B 1 4 D
�D kfrmID   (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H 3 2 B 5 D 7 F F - 3 1 A 3 - 4 F A 2 - 9 F 6 1 - E 8 0 D 0 9 C C B 7 4 C
�@ kfrmID   .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H 5 2 5 3 A D 9 1 - 6 2 5 9 - 4 2 7 1 - 8 A 2 8 - 5 0 E 1 F 9 E D F 4 5 E
�< kfrmID   44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H 3 9 9 C D 5 9 7 - 6 4 F A - 4 B D A - 9 1 9 A - A 9 0 2 0 8 5 2 A 7 A 8
�8 kfrmID   :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H 1 E C C 7 A 1 E - 3 6 5 3 - 4 C 2 2 - B A 0 8 - B 5 8 B 0 9 4 F 1 B D 7
�4 kfrmID   @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H 5 7 D 8 A 5 C B - 6 2 4 E - 4 8 9 6 - A B A 0 - C 5 6 3 F F E A 0 A 5 0
�0 kfrmID   FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H C C E 4 6 4 5 E - D B 4 3 - 4 E 2 D - B 3 2 3 - 8 B 7 7 C F 2 6 E E 6 4
�, kfrmID   LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H 4 C 1 3 2 2 3 E - 3 C 7 C - 4 B B D - B 4 7 2 - 1 5 A 9 F 7 F 7 0 4 C 4
�( kfrmID   RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H E E C 5 6 9 D 3 - E 6 A 8 - 4 8 1 F - 9 6 A 7 - 7 4 B 2 2 A 7 E 6 9 E 5
�$ kfrmID   XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H A 8 9 4 4 1 A 7 - 0 C 5 D - 4 0 E A - A 4 0 7 - D F 0 1 7 F 7 A 6 C E E
�  kfrmID   ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H 9 E C A 0 2 5 3 - C 2 B D - 4 0 4 A - A 9 6 D - 1 9 6 E 1 3 D 6 A 3 0 2
� kfrmID   dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H B 9 8 8 E 3 8 6 - 8 3 1 0 - 4 0 F E - 9 2 1 A - B F 2 9 F 5 3 C D 7 D 9
� kfrmID   jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H 0 C C 8 7 4 9 6 - 3 A 8 2 - 4 F F 2 - A A 7 3 - 2 2 4 E 7 E 2 2 F 3 5 E
� kfrmID   pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H 2 3 B C A 2 9 E - F D C 5 - 4 8 E 0 - 8 D 8 7 - 1 8 6 E 3 A C F 6 7 1 A
� kfrmID    vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H F 4 A A A 8 C 5 - 8 1 4 B - 4 D 0 F - B 7 A D - 5 0 7 8 F 9 4 A D A 8 9
� kfrmID  ! || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H 7 5 C 9 2 6 9 0 - C D 9 C - 4 C 2 B - 9 C 6 1 - 6 5 4 8 F 4 E A 2 E B E
� kfrmID  " �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H C 1 8 5 D F C A - 9 C C B - 4 8 3 D - A 9 B E - C 6 9 E 2 A 4 2 2 C 8 E
� kfrmID  # �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H E F 0 8 9 4 9 D - B 3 9 2 - 4 7 1 9 - 9 A B 4 - 6 3 1 E 1 4 D A E B 2 2
�  kfrmID  $ �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 D 3 1 8 A 2 E - B 6 B E - 4 D 3 3 - 9 9 4 C - A 7 3 C 1 7 C C 3 9 0 C
�� kfrmID  % �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 6 D 1 B 2 A 9 - 4 3 C B - 4 4 A 6 - 9 9 6 3 - 4 2 7 5 4 9 7 8 1 7 0 4
�� kfrmID  & �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 1 1 1 7 2 A 5 - C 1 6 B - 4 0 D E - 9 D 9 A - F C 3 9 E 9 1 C D 5 3 6
�� kfrmID  ' �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 3 7 9 9 F 6 4 - F F F 4 - 4 4 8 9 - 8 4 C B - 4 E 1 C 4 4 E E 8 E 4 0
�� kfrmID  ( �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 0 9 7 7 5 D E - F 9 2 4 - 4 4 7 C - B D 5 A - D 0 E F E 7 1 8 8 4 E 9
�� kfrmID  ) �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 5 4 8 3 6 E 7 - C 5 6 6 - 4 A F B - A A A 2 - D D E B F 5 A 4 1 C E 0
�� kfrmID  * �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 6 9 2 4 F 8 0 - B D D F - 4 F 8 9 - 8 C 6 9 - 5 C E 9 B 1 3 2 0 F 9 D
�� kfrmID  + �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 B 7 D F 5 6 6 - 7 6 7 D - 4 C D 6 - B 0 2 6 - F 2 C B 6 C A 6 6 9 C 3
�� kfrmID  , �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 3 1 F E 8 3 0 1 - A 1 B 3 - 4 9 1 F - 8 E 4 F - 8 7 B 8 8 E 4 4 8 B C F
�� kfrmID  - �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 6 5 9 3 F F 5 - A 2 1 7 - 4 6 1 4 - A F A 3 - B F D 0 9 E E 5 4 8 6 B
�� kfrmID  . �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 9 8 8 D 7 B 4 - F F C 5 - 4 3 6 0 - 9 7 5 D - 4 B E C 0 9 3 F 5 6 A 8
�� kfrmID  / �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F 1 7 A B 5 7 3 - 1 0 A B - 4 C 2 E - 8 8 F A - B A 1 F E 0 E 0 0 3 B 4
�� kfrmID  0 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 A F 1 F C E 1 - 2 3 C 9 - 4 7 0 A - A 4 A 4 - 0 3 0 9 0 4 6 2 8 C 4 8
�� kfrmID  1 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 0 1 2 E F 4 D - 9 6 5 0 - 4 3 0 A - 9 C 5 5 - 4 E 7 6 6 9 9 E C C 7 A
�� kfrmID  2 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A D 6 7 3 1 5 9 - F 7 3 D - 4 8 5 F - B 9 4 D - 6 7 7 6 C F 7 C B E 8 5
�� kfrmID  3 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 0 E 2 C D E B - C F B B - 4 A B 3 - 9 F A A - C 5 C 5 8 2 4 C D F 6 3
�� kfrmID  4 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F F C 1 3 1 0 7 - 3 6 C 2 - 4 B B 7 - A 1 6 0 - 2 4 3 6 F 2 F 0 9 C D 6
�� kfrmID  5 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 1 7 5 F 0 E 4 - 2 4 F 8 - 4 6 1 6 - 8 5 4 3 - 7 9 2 3 6 2 5 0 F 3 D 5
�� kfrmID  6 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 1 F 5 F 2 8 0 - F B 1 0 - 4 1 1 7 - 9 4 4 8 - 7 C 2 7 F 3 5 B E 5 A 1
�� kfrmID  7    ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 0 5 6 1 3 3 7 2 - 5 0 A C - 4 2 A A - A E F 5 - D 9 2 5 B D 5 0 1 8 D 5
�� kfrmID  8  ���� b��	��
�� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 1 F B 4 C 7 2 E - 5 3 E 8 - 4 4 6 0 - 8 0 6 B - 2 7 6 9 2 6 1 E 4 8 4 D
�� kfrmID  9  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 8 5 6 A E 1 9 C - 6 B 2 C - 4 F 8 C - 9 5 2 A - 5 B 9 9 2 5 2 7 C 6 B 0
�� kfrmID  :  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H E A 4 A A 6 0 C - 3 9 1 8 - 4 4 1 5 - 9 B 2 4 - D 2 0 1 6 1 0 F F A B E
�� kfrmID  ;  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 8 2 4 6 2 5 6 3 - 9 7 1 C - 4 4 1 9 - B 2 0 2 - A B F 0 3 3 B 2 8 B 0 2
�� kfrmID  <  �� �� b��!��
�� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �## H 3 6 5 F 8 1 6 3 - C 0 0 D - 4 7 8 D - 9 9 3 3 - 1 3 A 1 0 5 C 9 5 6 F C
�� kfrmID  = $$ %��&��% b��'��
�� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �)) H 5 5 2 D 2 E F 5 - 7 A B E - 4 D 9 6 - 9 B 5 B - 8 E 0 3 D 9 8 2 5 0 7 7
�� kfrmID  > ** +��,��+ b��-��
�� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev, �// H 1 B 7 0 7 3 3 7 - E 2 F 0 - 4 0 8 C - 9 0 D B - 9 6 B C 1 B D 3 C A 3 7
�� kfrmID  ? 00 1��2��1 b�3�
� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev2 �55 H 7 8 C 7 8 C 4 4 - 8 C 8 5 - 4 D 6 5 - 8 C 7 3 - 6 0 D D 7 D 4 A 0 C F 0
�� kfrmID  @ 66 7�8�7 b�9�
� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev8 �;; H C 0 1 2 E C B 7 - D F 1 E - 4 6 D C - B A 4 C - D 9 F 9 2 2 1 C D C D 1
� kfrmID  A << =�>�= b�?�
� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev> �AA H 2 B 2 B 2 4 5 1 - 0 3 4 B - 4 7 7 A - 8 E 2 E - 1 A 3 F 5 0 2 0 2 C 1 0
� kfrmID  B BB C�D�C b�E�
� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevD �GG H 1 8 9 8 7 B 3 4 - 4 6 D F - 4 F 7 6 - 8 6 A 6 - 5 7 D 8 7 5 7 8 0 5 4 D
� kfrmID  C HH I�J�I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrevJ �MM H F F 8 1 B D A 5 - A 1 2 3 - 4 B 2 8 - 9 3 D 8 - 8 3 A 1 1 E 8 4 2 B 4 7
� kfrmID  D NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H F A D 0 5 3 1 D - 0 9 A 5 - 4 0 8 4 - 8 7 5 0 - E E A 7 C 2 E A D 6 A A
�| kfrmID  E TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H 2 3 3 B C 2 4 A - 5 5 A 6 - 4 7 5 6 - 9 0 9 E - 6 2 C 6 4 0 D 3 5 3 4 2
�x kfrmID  F ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H 4 0 0 A C 5 9 3 - 4 3 C D - 4 9 C 9 - 9 C 0 D - E 3 D 9 6 1 4 9 0 0 6 2
�t kfrmID  G `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H 6 6 B 8 7 3 6 9 - 8 E F D - 4 E 8 E - B 8 6 4 - B D F 5 3 8 A 9 3 5 7 9
�p kfrmID  H ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H B F 9 7 C 8 5 D - 6 0 E D - 4 6 0 9 - 8 2 F D - 9 1 2 0 3 F 4 F 6 2 9 F
�l kfrmID  I ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H 9 4 F 4 5 0 3 3 - 8 E 9 3 - 4 F 5 E - 9 D 1 B - 1 3 9 3 5 C 6 F F E 9 0
�h kfrmID  J rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H 0 3 0 A 5 2 E A - 1 0 B 0 - 4 3 F 1 - B 0 4 E - 2 A 2 3 9 F 2 B F 2 B D
�d kfrmID  K xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H 6 D F 5 2 0 A 5 - E C 0 F - 4 D 5 5 - 8 8 8 0 - F 8 1 2 D 3 9 0 D 9 4 D
�` kfrmID  L ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 4 5 A 5 C 7 D A - 1 7 B 3 - 4 6 3 A - 9 B 7 8 - 4 A 7 6 1 D E A 3 1 7 8
�\ kfrmID  M �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H 9 7 0 5 2 C 2 C - 9 4 D B - 4 0 5 8 - A B 2 5 - 6 4 1 0 6 2 7 7 E 1 F A
�X kfrmID  N �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H D 2 8 3 C 8 4 0 - D 7 6 A - 4 6 4 D - A B 7 D - 3 6 A 0 4 C A 5 8 9 6 D
�T kfrmID  O �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H 1 D B D 6 3 A 2 - 3 C 1 D - 4 8 A 5 - B 2 2 4 - D 6 5 D 2 5 8 D 9 8 6 0
�P kfrmID  P �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H 1 8 C D 9 7 C 2 - A 8 B 0 - 4 9 0 9 - A D E 6 - 0 5 0 1 8 1 9 1 E 4 A C
�L kfrmID  Q �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H 2 7 D F F 4 9 6 - 1 F 2 8 - 4 0 D E - A D C 6 - 4 F 4 4 A A E 2 2 2 C C
�H kfrmID  R �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H A 9 C 4 B 0 A 0 - 1 D C 1 - 4 B A 6 - 9 0 0 C - 9 9 5 2 3 6 B 1 9 2 3 6
�D kfrmID  S �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H 9 3 D F B 4 4 F - 0 D C 0 - 4 2 2 D - 9 D A 3 - 4 F F B B B 8 3 B 9 5 C
�@ kfrmID  T �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H 0 9 2 2 C 2 F 4 - E A 7 C - 4 B C 6 - B 4 B B - 0 D 1 F B 7 2 9 A 1 A 9
�< kfrmID  U �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H B B 3 C E B 3 9 - 3 B F 5 - 4 8 C A - 9 D 8 3 - 1 0 7 1 5 A 0 7 4 1 2 3
�8 kfrmID  V �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H C 9 6 3 2 4 D 5 - 3 5 F 2 - 4 4 3 2 - 9 5 9 6 - 3 9 3 7 1 9 3 6 7 F D 9
�4 kfrmID  W �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H 0 3 3 6 B E D 3 - A C 0 E - 4 B D 6 - 8 F 7 1 - 5 2 3 C 3 0 6 E 4 1 A D
�0 kfrmID  X �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H 6 1 C 2 8 5 D 2 - 8 3 8 F - 4 6 3 0 - 8 4 9 9 - E A A 3 7 2 9 E 3 3 F 2
�, kfrmID  Y �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H D E 9 2 B C 6 0 - A 9 2 A - 4 5 F D - 9 6 9 1 - 6 9 9 4 C 9 D 2 1 8 7 9
�( kfrmID  Z �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H F 7 0 C 0 4 E B - A 3 6 7 - 4 B 3 F - B 2 4 B - 8 6 E B 6 4 A 3 B A 4 A
�$ kfrmID  [ �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H 8 D 3 0 A E C 0 - D 7 F 7 - 4 0 2 D - B 7 7 3 - 0 3 1 9 2 6 2 B 7 A 6 7
�  kfrmID  \ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D 3 6 6 8 D 5 2 - E 4 9 E - 4 A 4 1 - A 5 B A - 4 E 0 1 6 B D 9 F 2 5 C
� kfrmID  ] �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 4 D 1 9 2 F C - D 5 2 3 - 4 1 E C - 8 D B 7 - 9 C 0 0 9 E 7 0 4 3 9 9
� kfrmID  ^ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 9 2 C E 2 5 8 6 - 7 A 9 7 - 4 8 F 4 - A 1 F B - 9 4 C 7 1 3 C 3 B 0 D 2
� kfrmID  _ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E F C 5 5 A 1 0 - B D E 1 - 4 4 D D - B 8 4 C - 4 4 C 6 8 E F 9 0 9 E 0
� kfrmID  ` �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H 0 D 4 E 5 A F 5 - 8 7 8 3 - 4 C 4 F - 8 6 0 4 - C 7 D 0 9 A 3 8 8 1 A E
� kfrmID  a �� ��	��� b���
� 
wres� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� � H 5 6 1 7 C 4 A 1 - 2 7 4 A - 4 4 4 1 - 8 E E 2 - 9 2 4 6 1 D 3 4 6 7 B 5
� kfrmID  b  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 7 E 5 D 5 8 9 4 - 1 6 0 1 - 4 4 9 4 - A B B 6 - E C F 5 5 6 1 F 1 0 E 7
� kfrmID  c  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H B 6 5 B 7 7 5 3 - 1 3 1 4 - 4 7 8 2 - B 5 1 3 - 2 1 3 C 3 3 9 B 6 1 7 8
�  kfrmID  d  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 3 0 5 8 D 5 2 E - 3 5 B B - 4 8 9 1 - B 5 7 8 - 9 D E 6 5 C 7 D 8 2 7 9
�� kfrmID  e  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H E 0 2 7 E 8 E B - 0 C A 9 - 4 C 0 2 - 9 E E F - 6 0 C 8 1 5 C 7 7 D 7 7
�� kfrmID  f  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 5 1 5 E 7 7 7 B - E F 9 C - 4 B 3 D - B 8 9 5 - 9 3 1 9 2 A 0 2 A 3 6 E
�� kfrmID  g    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H 3 9 8 2 7 D A F - F 7 A 7 - 4 2 7 4 - A 0 E 6 - 0 4 6 C A 7 D 3 3 6 E F
�� kfrmID  h && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H C C F 4 2 B 4 9 - 2 8 B 0 - 4 C 6 F - B C 1 B - 0 2 A 5 F 5 5 2 7 B A 9
�� kfrmID  i ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 9 B D F 6 5 D 0 - 6 E 6 C - 4 5 D 2 - 9 8 A 0 - F B F 2 4 6 F 0 F A 6 F
�� kfrmID  j 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H 8 C E 7 D D 0 5 - A F A 6 - 4 4 9 1 - 8 8 A B - C B A B 1 2 F B 8 E 5 2
�� kfrmID  k 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H B D B 8 E 5 3 2 - 2 0 0 9 - 4 D 7 A - 8 D A 8 - A B 2 F 6 B 3 4 2 1 8 A
�� kfrmID  l >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H 5 2 C 6 B D 8 D - 6 C 9 3 - 4 2 B C - B 0 3 7 - F A 9 E 3 5 D 8 2 A B C
�� kfrmID  m DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H 4 E D F B C F B - 8 9 F 3 - 4 3 D 7 - B 8 E 2 - 7 0 F A A 6 8 7 C 5 C A
�� kfrmID  n JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H 4 3 0 7 3 B 6 7 - 4 7 4 A - 4 5 7 5 - A 3 1 0 - 9 D 2 B E 0 3 5 E 9 B D
�� kfrmID  o PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H E 6 4 B D 0 A 2 - F 2 E C - 4 6 A B - B 6 C 8 - 2 6 9 5 D A 5 F 7 4 2 4
�� kfrmID  p VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H 6 A 9 7 B 8 6 A - 8 9 E 2 - 4 4 5 9 - 8 4 F 5 - 1 4 B 0 9 5 0 8 F B 3 6
�� kfrmID  q \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H B 2 B 1 4 1 9 7 - E 4 E 7 - 4 8 B F - A C A 8 - 6 E C C 6 9 A 4 2 4 E 6
�� kfrmID  r bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 4 3 5 F 6 C F D - 7 2 9 3 - 4 F 0 4 - 8 0 D 9 - 5 2 0 6 B 9 C 9 C 6 1 F
�� kfrmID  s hh i��j��i b�k�
� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrevj �mm H 2 7 B 0 3 0 D 2 - 5 F B 0 - 4 D 0 B - A A D 8 - 8 3 F F 1 6 5 B C 6 1 0
�� kfrmID  t nn o�p�o b�q�
� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevp �ss H C 5 2 7 D 8 2 E - A 5 E 1 - 4 6 0 6 - 8 3 6 C - 7 3 7 3 A 6 E 1 2 1 7 3
� kfrmID  u tt u�v�u b�w�
� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevv �yy H 0 E 3 C B B 5 7 - 3 8 9 E - 4 B 5 5 - A C 7 B - 0 D 8 F 0 B 8 4 F 5 6 2
� kfrmID  v zz {�|�{ b�}�
� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev| � H 8 F 5 0 7 2 3 A - F F E 6 - 4 D B F - 9 0 1 A - E D F 6 3 1 6 5 1 8 F 2
� kfrmID  w �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 3 3 E F 2 8 1 - C 2 A 2 - 4 C 8 D - A 2 1 F - A 1 0 A D 3 F 6 0 2 B 8
� kfrmID  x �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E 3 7 0 D D B 3 - A 5 D 9 - 4 D 6 A - 8 8 A 9 - A B 7 0 A 7 D 2 A E 7 E
� kfrmID  y �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 9 C D F 1 2 1 C - C 1 0 4 - 4 0 D 8 - 8 D B 0 - 9 E F 2 E F 4 8 E 1 4 1
� kfrmID  z �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 9 F 2 F C E 5 6 - E 0 2 0 - 4 C 1 B - 8 6 8 E - 2 B F 1 5 5 2 4 A 9 2 A
� kfrmID  { �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D F D 6 D F D 6 - C 4 F 4 - 4 3 4 E - B A A 8 - C 4 6 8 6 7 7 D 9 B E D
� kfrmID  | �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 1 8 E D 7 1 2 7 - 3 6 F 8 - 4 1 2 3 - 9 3 2 0 - 6 6 1 D B 9 9 5 4 F F 0
� kfrmID  } �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D 9 B 8 9 C 0 6 - B 3 B 7 - 4 0 8 4 - 8 E 7 3 - 4 7 B E 2 A 0 9 7 D 2 C
� kfrmID  ~ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 2 C 3 F E E 9 D - D E D 5 - 4 1 8 4 - A 4 A C - 2 E 1 A 1 2 0 1 D A 4 4
� kfrmID   �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H F 7 C 2 9 4 5 0 - B E 9 A - 4 A D 7 - 8 3 1 D - D 1 8 F 4 A 6 4 C B 7 A
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D 3 2 1 B B 8 C - F 1 4 C - 4 A 6 D - A 0 6 7 - 4 C F 1 D 3 C 3 4 C E 4
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 1 C 0 1 E B 4 8 - 9 A 9 9 - 4 B B F - 9 4 B D - 8 0 A 4 5 B F 1 0 1 6 A
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 9 A 4 C 4 2 1 D - C D 0 8 - 4 9 1 8 - 9 7 4 4 - 4 2 F 7 F B 3 D 6 7 1 B
� kfrmID  � �� ����� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev� ��� H 2 D 5 5 3 5 6 4 - 3 A F 4 - 4 D 4 9 - B 7 7 8 - 0 1 7 F 0 9 A F 7 3 8 A
� kfrmID  � �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H 7 1 8 2 5 4 7 4 - E 0 3 C - 4 E E 9 - A D 8 E - 4 0 2 9 7 F 0 4 9 6 1 8
�| kfrmID  � �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H 4 8 8 2 B 9 0 4 - 6 F B E - 4 5 9 5 - A E 6 E - 7 E E E 7 4 0 3 3 C D 7
�x kfrmID  � �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H E E C C 5 5 A 6 - B 3 5 3 - 4 C 9 4 - A 1 9 C - E 5 9 C C 9 D 5 8 4 2 C
�t kfrmID  � �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H 1 A F 3 F 5 0 7 - 3 6 5 F - 4 2 1 E - B 2 9 6 - 3 2 E 4 7 D A 9 9 B 2 3
�p kfrmID  � �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 0 B 1 F 3 5 0 6 - 8 E B 1 - 4 1 1 F - 8 9 6 3 - A 5 E B 8 0 0 B D 6 1 E
�l kfrmID  � �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H 6 5 F F F 5 4 4 - A 9 7 C - 4 9 C 7 - 9 C 4 C - 9 3 0 E A 9 C C C 5 0 C
�h kfrmID  � �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H A C 8 6 5 D D 8 - 1 2 F A - 4 C C F - B C 9 5 - 5 8 8 C 0 1 8 2 4 1 9 8
�d kfrmID  � �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H 4 0 A B 1 6 E F - 4 5 3 3 - 4 8 3 B - 8 7 4 D - 4 0 6 6 3 9 6 D E 9 E E
�` kfrmID  � �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H D 0 1 E 0 E 3 C - 9 8 B 1 - 4 E 9 8 - B 6 5 9 - 3 9 F 2 D E B B 4 3 9 8
�\ kfrmID  �  �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H 2 0 3 B D 1 0 7 - 4 6 E D - 4 C 0 1 - B 5 F A - 0 2 7 1 C E 2 1 4 3 1 0
�X kfrmID  � 

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H 2 6 B 2 8 2 B 8 - A A F F - 4 E 2 1 - A F 4 A - F 2 4 C 2 A 6 9 3 C 6 4
�T kfrmID  �  �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H 0 0 C 3 4 3 0 2 - D F 3 7 - 4 9 2 E - B A 1 B - 5 2 9 0 9 E B 3 4 1 D 1
�P kfrmID  �  �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 8 D B A 6 F 1 F - 6 D E B - 4 7 6 F - 9 2 7 7 - D D E 0 D E 1 D C 4 0 F
�L kfrmID  �  �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H 7 B D 0 A 2 0 5 - F F C D - 4 2 7 8 - B 9 0 3 - 0 D C 9 7 1 F C 8 2 E 5
�H kfrmID  � "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H A 1 A D E 3 5 C - C C 8 C - 4 0 C 6 - 8 D C B - 9 8 D 5 2 C 7 D B 4 C 3
�D kfrmID  � (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H 7 7 A 1 3 2 4 0 - 2 9 6 6 - 4 3 2 F - 8 5 1 6 - A C 2 6 5 7 9 2 1 1 7 B
�@ kfrmID  � .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H 6 5 E A 0 4 D 8 - 8 0 E 9 - 4 6 4 6 - 8 3 2 C - 8 1 B F 4 1 4 3 E 7 0 1
�< kfrmID  � 44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H 9 A 3 F 6 1 0 7 - 2 F B C - 4 7 F 7 - 9 9 B 9 - C 3 8 6 1 A 8 F A 4 A D
�8 kfrmID  � :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H 3 5 A A D 7 9 E - 8 9 B 5 - 4 D 3 5 - B 3 C 6 - A 8 9 8 4 B 0 C C 1 0 C
�4 kfrmID  � @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H D 7 9 9 5 5 3 B - 3 0 7 6 - 4 B 2 6 - B 6 7 C - 4 A 5 F F 0 E 6 2 C 1 E
�0 kfrmID  � FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H 4 E C A 5 8 0 6 - F 5 F 0 - 4 B 1 7 - A F 4 F - 2 1 F A 3 7 A C 4 8 7 E
�, kfrmID  � LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H F E F 5 F C E B - 1 8 B 2 - 4 F D D - B 6 3 9 - 5 B 2 A 0 4 A C 1 9 7 1
�( kfrmID  � RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H F 0 C B A B 0 9 - 9 8 3 6 - 4 D D 9 - 8 E 7 D - 5 5 C A 8 0 6 1 0 9 E 9
�$ kfrmID  � XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H C 9 9 0 4 C 4 2 - 8 5 D 3 - 4 7 7 E - 8 A D D - 2 7 4 E E 9 A 8 1 D C 5
�  kfrmID  � ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H F 2 B F 0 1 0 7 - 2 F F 1 - 4 1 7 5 - A 3 5 D - 2 7 4 6 6 7 B 0 8 C 4 1
� kfrmID  � dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H 3 C 4 8 D 9 C F - F E 6 2 - 4 2 9 C - A 0 4 A - 3 9 4 6 F 8 2 0 3 2 2 7
� kfrmID  � jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H D F 5 1 3 1 8 8 - 7 7 D 6 - 4 C 8 4 - 8 5 5 0 - 9 8 D A A 2 6 7 C B C 3
� kfrmID  � pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H 3 0 D 1 3 D 9 5 - C 7 2 3 - 4 5 F 9 - A F 2 C - D 4 F 9 5 7 C 3 A C 1 0
� kfrmID  � vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H 0 7 7 C 4 6 9 6 - 5 3 3 3 - 4 7 8 F - B 1 4 1 - D 2 5 F C 7 A B 9 3 7 A
� kfrmID  � || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H 6 9 4 9 6 3 1 3 - F 9 A F - 4 1 1 2 - 9 A 4 F - 1 6 B 9 4 7 4 8 4 1 4 2
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H A 2 3 6 D E 1 E - F A A 8 - 4 3 D F - B 1 F 7 - 6 E 8 F 2 5 A 2 6 0 0 A
� kfrmID  � �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H 5 E 2 8 1 E 5 2 - A 7 5 E - 4 E E 1 - 9 A 9 E - 4 C 5 5 A C 6 E F 0 2 5
�  kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B E 9 C 5 5 6 7 - D D E F - 4 0 8 8 - 9 7 F 2 - 1 8 9 3 D F 9 8 A 1 7 A
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 4 3 8 F F 1 2 - 8 2 5 6 - 4 1 7 0 - A B 0 1 - 2 D 6 6 A A A 9 8 D 7 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 6 F A 0 F D 0 - 9 A 1 F - 4 E 9 2 - B 2 2 6 - 4 D 0 3 B B 5 7 2 5 7 4
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 3 1 A F 4 F B - 2 C E E - 4 9 4 2 - B 5 C B - 5 3 F 1 9 0 3 F 8 6 0 9
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 8 1 B B 5 0 1 - 8 E F 2 - 4 0 6 6 - 8 2 7 3 - 3 9 8 C 3 8 4 3 3 6 C 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F E 1 B 2 1 0 2 - E D 5 8 - 4 C 5 2 - A 5 5 5 - C 0 9 2 1 3 9 8 F 3 9 6
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D 3 0 7 4 F 9 9 - B B 5 C - 4 5 6 9 - A B 3 D - A F 1 1 2 5 B 6 1 E 7 4
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 1 6 9 B 0 5 4 - 3 F C 4 - 4 D 2 6 - 9 E 5 8 - 2 E D 1 B 4 D 9 A C F F
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E 8 8 A 7 4 9 2 - 1 4 7 C - 4 D 5 6 - B A 4 4 - 4 C 4 4 E A 1 2 1 B B 0
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 E 0 4 3 4 3 5 - F 6 F 3 - 4 7 E 1 - 8 0 C D - D F D C 8 4 E D 2 F 8 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E B 1 6 D 5 B 4 - F E 4 E - 4 8 F 0 - B B 3 F - E F A 2 9 3 C A 2 2 8 A
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 1 C D E 5 A C - 1 1 9 2 - 4 F A D - B 3 2 9 - 7 E E 3 B E 7 8 2 6 9 4
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 3 6 5 B 6 1 F 1 - E F 8 5 - 4 2 F 1 - B 0 A 5 - 0 5 E 6 9 B E 2 F B 5 3
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 9 3 6 F A 5 4 - C F 2 4 - 4 3 E C - 8 3 B E - F 1 4 7 D 5 3 B 8 D E B
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E D 8 E 7 4 C 7 - 3 9 6 0 - 4 E 9 C - 9 7 1 D - 8 6 4 B D D 8 B D F 1 0
�� kfrmID  � �� ������� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev� ��� H 8 5 9 A 3 E E 8 - 1 E F 3 - 4 6 9 A - 9 8 2 9 - 5 C 6 0 3 2 E E 7 E 3 C
�� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 7 5 4 7 1 0 2 4 - 4 A A 5 - 4 6 3 3 - B 4 C 0 - 3 6 2 5 F 3 9 5 9 6 F 4
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H C 3 F 0 C 4 4 3 - F 8 1 4 - 4 2 B 0 - A E D A - 6 E 3 E F E A 4 9 6 A 6
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H C 1 E 4 B 2 6 3 - 8 3 8 6 - 4 C 4 D - A 8 6 6 - B 3 6 9 8 A B 4 3 6 1 6
� kfrmID  �    �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H F 6 C 9 3 7 6 B - 7 3 8 1 - 4 F 3 4 - 8 3 4 D - A 9 C 0 E C 0 D 0 B 4 E
� kfrmID  �  �� b�	�
� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 6 E 0 B 8 B 5 E - 8 9 6 D - 4 2 0 6 - B 4 A A - 6 4 1 5 5 F 8 3 9 E F 0
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H A 5 2 8 1 C 3 1 - 7 E 2 0 - 4 8 B D - A F 6 F - 7 C 8 7 6 0 4 E C F D 9
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H E 7 0 3 3 5 5 A - 8 3 A B - 4 7 D 4 - A 5 9 B - 2 1 C 5 D E D A E D 9 2
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 2 E B 0 2 B 2 7 - C 5 C 5 - 4 4 A 6 - A 2 7 F - 9 0 8 5 4 8 7 8 4 E 1 0
� kfrmID  �  � � b�!�
� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev  �## H 2 3 7 A 5 1 C 4 - B 8 6 5 - 4 0 1 3 - A 6 3 A - 3 0 E 5 D 4 5 5 0 5 5 9
� kfrmID  � $$ %�&�% b�'�
� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev& �)) H D 3 6 0 2 1 A 9 - 4 5 B 9 - 4 8 1 4 - 8 8 8 A - 3 3 6 0 A B 5 1 9 5 F A
� kfrmID  � ** +�,�+ b�-�
� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev, �// H E 9 4 F 8 5 E C - 0 D A 6 - 4 5 2 4 - A A 1 E - 1 8 8 0 F E 0 2 7 5 D 5
� kfrmID  � 00 1�2�1 b�3�
� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev2 �55 H 0 9 7 1 3 F A D - C 2 B 2 - 4 1 7 C - 8 C 9 7 - A B C 8 1 6 2 C A 2 D 4
� kfrmID  � 66 7�8�7 b�9�
� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev8 �;; H D C A 9 C B C C - 2 7 3 D - 4 9 F 4 - B 8 3 0 - D D A 9 B A 3 F 0 B 2 3
� kfrmID  � << =�>�= b�?�
� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev> �AA H 1 1 6 B E 6 B B - 1 2 F 7 - 4 6 8 C - 9 B 3 0 - F 0 9 0 D 0 2 3 A E F F
� kfrmID  � BB C�D�C b�E�
� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevD �GG H 2 3 7 8 D 2 F F - C 1 4 3 - 4 F D 3 - 9 2 0 3 - D B 4 E 8 7 D F B 1 F 1
� kfrmID  � HH I�J�I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrevJ �MM H 0 F E 4 8 7 B B - 7 C C 1 - 4 B 0 7 - A E A 4 - 6 B 1 F 9 0 4 C 0 4 D 1
� kfrmID  � NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H 8 6 4 7 A 3 F 6 - 7 6 0 2 - 4 2 7 F - A 3 4 5 - B 6 C F D 8 5 1 9 2 5 B
�| kfrmID  � TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H E 7 C B A 0 0 4 - A C 0 5 - 4 8 8 8 - A F C 3 - D 2 0 8 5 8 2 1 0 E 4 9
�x kfrmID  � ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H 8 8 B C C F D 5 - 8 7 0 1 - 4 C 7 A - A 2 A 0 - 7 8 B 4 F 6 C 6 6 0 4 4
�t kfrmID  � `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H A 3 2 1 6 1 4 6 - A 9 F 2 - 4 4 8 C - A B 6 0 - 2 A 2 B 7 2 C 3 9 E 2 A
�p kfrmID  � ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H 0 E 5 2 6 A 5 B - D C C 1 - 4 2 3 6 - B 8 D 1 - 4 0 5 6 3 B 2 8 F A 2 8
�l kfrmID  � ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H 3 4 6 C 7 3 D F - E 7 5 D - 4 1 E F - B C 6 7 - 9 D F 2 D 3 7 3 D 7 D B
�h kfrmID  � rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H D 3 0 C F 6 F 0 - 0 7 9 F - 4 F B 9 - 9 0 3 B - 1 C 0 D 3 6 E 5 C 9 3 0
�d kfrmID  � xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H 2 D E 7 E A 8 3 - B 9 A A - 4 8 1 9 - A 0 7 4 - 4 5 8 B B 3 F 0 9 8 C D
�` kfrmID  � ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 2 F 0 D 3 B 7 C - A E F F - 4 D 4 6 - 8 6 9 E - 3 D F 2 0 E B 9 E 2 4 4
�\ kfrmID  � �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H 0 D E 5 0 7 8 D - 8 F A 9 - 4 1 B 6 - B 0 7 4 - 8 1 A 3 C 0 C 2 E 3 8 7
�X kfrmID  � �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H 5 C 2 4 D 8 6 E - 6 F F 0 - 4 C 9 1 - 9 8 A 5 - 8 A 8 0 5 5 9 7 8 A A 0
�T kfrmID  � �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H B 8 5 8 5 1 4 F - D C 0 B - 4 8 D 4 - 8 2 1 6 - 2 2 B 1 6 A 7 C 8 8 1 1
�P kfrmID  � �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H D 0 4 6 E F B 1 - 6 0 1 2 - 4 0 8 9 - B C D E - 6 3 1 D 1 7 4 F 2 D 0 8
�L kfrmID  � �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H 3 7 8 5 6 1 0 5 - 4 A B A - 4 1 5 7 - 8 0 C 9 - B 3 5 D E B C 8 5 1 8 F
�H kfrmID  � �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H 6 D A 8 9 4 1 2 - 3 9 8 7 - 4 2 C 6 - 8 7 C 9 - 6 B 6 C A 9 5 2 0 F 0 8
�D kfrmID  � �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H D 2 9 4 9 9 9 4 - 6 5 6 C - 4 0 A D - A C C 5 - 1 8 0 8 4 4 E 3 B 3 B 1
�@ kfrmID  � �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H E F F 0 8 5 C 5 - F 7 6 7 - 4 D A 5 - B 0 E 3 - E A 2 F B 5 1 D 7 B 9 8
�< kfrmID  � �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H E 6 F 0 7 A D 2 - 9 9 6 4 - 4 F A A - B 0 D 9 - 0 9 1 1 2 8 3 B E D 0 C
�8 kfrmID  � �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H C 4 4 3 F A 6 E - 5 D B A - 4 8 6 E - 8 8 6 D - 8 B 5 8 0 6 1 0 0 0 1 E
�4 kfrmID  � �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H D E 1 1 C 7 2 D - 0 3 5 0 - 4 7 B 0 - 8 6 8 7 - 2 3 A 7 E 9 7 C 0 4 B 5
�0 kfrmID  � �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H A D 5 9 F 9 6 2 - 9 9 2 6 - 4 9 1 4 - B B 0 7 - 7 1 2 0 7 2 F 2 9 7 7 7
�, kfrmID  � �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H 0 A D 4 7 D 2 5 - 6 D B 2 - 4 1 5 1 - A 5 9 F - 5 B 6 4 8 C D 5 8 F 2 5
�( kfrmID  � �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H E 3 D 9 F 4 F 5 - 8 2 F 7 - 4 5 B 3 - 9 3 A 6 - E C 6 7 A 1 7 6 E 6 4 4
�$ kfrmID  � �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H E 3 7 2 0 A D 5 - 2 5 5 A - 4 D 2 9 - 8 A D F - 4 8 F C 3 0 D C 8 F F 9
�  kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 0 D 4 9 3 F E - 5 D C 0 - 4 3 F 7 - 8 5 7 C - C 4 F 2 F 7 9 0 8 7 A 2
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 8 E 3 D 1 2 A F - 8 D 5 B - 4 C 4 0 - A 7 6 5 - 8 0 7 8 E 6 C 6 2 1 A 1
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 6 5 5 1 5 3 6 8 - 0 3 8 3 - 4 F 1 7 - 9 E 7 F - 5 9 5 5 7 E 5 D 5 C D 2
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D B 5 5 7 D 2 9 - D 4 6 6 - 4 1 A 9 - 8 8 6 3 - C B A 1 9 2 F 6 1 2 8 3
� kfrmID  � �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H 3 B 9 8 B 2 2 3 - B 6 2 4 - 4 B 5 0 - 8 C 3 6 - 0 0 C 1 8 0 7 1 3 A 4 E
� kfrmID  � �� ��	��� b���
� 
wres� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� � H D 2 D B E A A C - F 3 0 F - 4 A 9 C - 9 9 0 6 - D B D 5 B 2 F 0 C 6 B F
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H E 3 F 3 5 4 7 6 - 8 A 4 1 - 4 1 C 6 - B 6 4 D - B D 0 A 9 7 E A 5 B 8 B
� kfrmID  �  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H E 9 E 7 3 1 D 5 - B 2 8 8 - 4 C 4 9 - B 8 C B - 7 B F 9 9 4 D B B 0 F 7
�  kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H F 3 1 9 3 D 1 E - C 7 5 4 - 4 8 6 2 - 8 0 9 2 - A 9 8 8 5 D 3 D 9 A 5 3
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 7 E C 4 6 5 D B - 4 8 B C - 4 4 8 D - 9 8 E D - A 4 A D B 9 3 C E 9 6 F
�� kfrmID  �  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H C 6 A 5 C 3 D B - 2 C 6 5 - 4 5 4 3 - 8 3 1 8 - B D 8 B 7 D 3 D A A 0 E
�� kfrmID  �    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H A E 6 6 E E E 1 - 2 B F 0 - 4 B 9 D - B C 4 C - 1 B 7 4 5 8 4 B 5 3 7 1
�� kfrmID  � && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H 8 0 8 C E 2 0 4 - B C 7 7 - 4 1 F 3 - A 4 3 2 - E E 1 8 4 B B E E 4 F 1
�� kfrmID  � ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 9 9 0 E 2 3 7 1 - E 0 B A - 4 7 B 1 - 9 8 0 B - 5 4 7 0 7 4 F A 0 3 5 9
�� kfrmID  � 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H E 2 B F 3 4 1 B - 3 3 F 4 - 4 5 6 3 - A 2 2 2 - 0 8 C 6 0 7 0 B 7 5 1 A
�� kfrmID  � 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H 8 9 0 E 9 3 C F - F 6 6 9 - 4 F 4 7 - B 5 4 9 - C 1 F 9 A D 9 C 4 9 8 1
�� kfrmID  � >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H B F 0 5 9 4 5 9 - E A 4 8 - 4 1 0 4 - 9 C 6 9 - 9 A E 7 0 0 7 0 E 9 8 F
�� kfrmID  � DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H 9 D 0 0 0 2 9 C - E 3 E 5 - 4 D B 3 - 9 F 7 8 - A E B 3 8 4 9 E E 4 6 7
�� kfrmID  � JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H 6 7 F 9 E 5 7 8 - B C 7 A - 4 3 B 7 - B 1 E E - E 4 D B B 2 7 B F D 6 4
�� kfrmID  � PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H 7 5 1 B 7 C 8 3 - 5 2 0 0 - 4 1 A 6 - 8 6 0 F - 2 D A 9 9 F 3 5 3 6 8 4
�� kfrmID  � VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H A 0 B 1 8 6 6 F - E 5 1 7 - 4 3 4 5 - 9 6 E E - 4 C 0 1 9 4 4 D 9 6 7 B
�� kfrmID  � \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H 6 5 2 9 5 D A 4 - 9 D 5 C - 4 D B E - A 7 F 4 - 7 2 4 A 8 6 9 1 8 F A 8
�� kfrmID  � bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 4 2 1 0 4 8 7 0 - 4 E 8 3 - 4 E 4 4 - B 9 1 6 - 7 E C 0 3 A 0 1 1 F 2 9
�� kfrmID  � hh i��j��i b�k�
� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrevj �mm H 2 2 3 A F 0 B F - A 8 B 2 - 4 B 6 8 - A F D D - 2 F 5 4 6 A 2 C 3 9 B D
�� kfrmID  � nn o�p�o b�q�
� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevp �ss H 0 6 0 3 2 3 A 3 - E E 9 8 - 4 4 7 B - 9 4 0 F - B 3 2 7 6 4 3 7 F 6 E 4
� kfrmID  � tt u�v�u b�w�
� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevv �yy H 0 6 5 3 C A D 8 - B D 5 7 - 4 3 0 F - A 0 5 0 - F E A B A 2 2 A D 6 3 7
� kfrmID  � zz {�|�{ b�}�
� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev| � H F 1 B 2 2 8 4 B - 6 F D 3 - 4 E E E - 9 4 1 0 - A 0 D 8 7 B 5 3 1 B D C
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 B 8 3 5 B D 5 - A 3 F B - 4 9 0 4 - B 6 1 B - 7 6 2 7 8 0 6 B C F 3 5
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 6 B C 0 F A 8 D - 0 A B B - 4 D 9 8 - B E 8 1 - 4 B 3 2 B E 2 5 2 6 D D
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 F B 3 2 1 6 4 - 7 5 C 1 - 4 B B 2 - 8 F 5 5 - 5 A E A 2 F 7 2 4 E 9 C
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H F 4 D D 5 4 1 7 - 2 1 5 B - 4 3 4 4 - 8 2 A 1 - D 4 C C E 7 A 9 9 3 A 2
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 4 A 6 D 3 C 1 - D 2 6 5 - 4 5 0 C - A D 1 2 - 1 E E 1 6 F F 7 A E 8 D
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D 2 6 9 B 6 4 3 - 1 3 8 D - 4 D 4 1 - A 1 F B - 1 F E A 9 C C 6 9 7 D C
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 E 5 3 4 8 9 A - B C 5 0 - 4 1 B A - A A B 0 - 7 6 2 4 8 D A 4 0 4 D 0
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 7 1 0 8 1 C 6 F - 4 1 7 4 - 4 8 0 E - 9 6 B F - 0 A 2 2 9 6 6 F 8 2 F A
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 7 6 0 A 8 7 4 7 - 0 F 3 5 - 4 2 5 9 - 8 5 1 7 - A 0 0 4 F D 5 E 1 0 0 4
� kfrmID    �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 9 0 3 7 E 0 5 F - 0 0 4 6 - 4 D 8 D - B 8 C 5 - 2 F 2 3 5 D D 3 8 5 8 5
� kfrmID   �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 3 2 5 E 4 4 6 1 - A B 3 6 - 4 9 E 3 - 8 3 E 4 - 8 E 6 0 F C B E 7 3 4 8
� kfrmID   �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H A 1 E 7 6 C 6 2 - A F 8 8 - 4 0 1 4 - 8 5 4 F - 3 F 0 8 F B 2 9 2 1 2 7
� kfrmID   �� ����� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev� ��� H 4 7 1 9 B 2 1 9 - A 6 F 6 - 4 0 0 0 - A 1 1 A - 8 9 4 E 9 B A 7 0 D 1 D
� kfrmID   �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H 7 A E 2 D 9 B C - E 8 E 4 - 4 1 F 0 - A 4 6 1 - 7 6 1 1 D 5 0 9 D F B A
�| kfrmID   �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H 9 1 3 1 9 9 8 F - 7 4 7 9 - 4 0 D 8 - 9 B 6 F - A 3 F B 8 A 6 2 D 6 1 0
�x kfrmID   �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H 5 D 2 A C 3 2 8 - 0 E 6 D - 4 7 9 D - A E D E - 7 7 7 1 3 A 1 4 4 5 A 0
�t kfrmID   �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H 8 2 F E C A 4 1 - 7 F 6 E - 4 B 3 2 - B 1 D E - E 4 0 1 A 9 F E 7 A 8 7
�p kfrmID   �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 8 7 0 3 6 6 6 F - 3 A 9 3 - 4 A 3 9 - A 5 7 7 - 0 2 0 8 1 A 5 B D 9 E B
�l kfrmID  	 �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H 8 1 4 B 3 B 1 E - A F E E - 4 3 0 3 - 8 7 0 7 - B E 8 0 A 3 8 5 F A 1 F
�h kfrmID  
 �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H A 9 C B 7 5 E 8 - 5 D 5 5 - 4 6 F 3 - B 4 A 3 - D 0 A 7 3 0 9 6 E D 8 F
�d kfrmID   �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H D 9 D 2 C 0 A 8 - 1 1 D A - 4 C B E - 8 B A D - 3 D 8 9 F F 4 6 D E 2 8
�` kfrmID   �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H 1 C 5 D 4 D E F - F 5 7 9 - 4 8 7 3 - 9 5 6 4 - 2 D A 8 7 D E A C 5 7 4
�\ kfrmID    �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H 9 8 C 9 6 C 0 B - 5 5 2 A - 4 6 7 E - B D 3 B - 8 4 0 D 2 D E 0 7 5 E C
�X kfrmID   

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H E 4 1 6 3 B C E - B 1 8 A - 4 E 7 B - B 9 3 E - 4 F 9 0 9 2 9 B 1 6 1 9
�T kfrmID    �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H F 6 B 6 9 6 5 7 - 5 C 4 8 - 4 9 D 9 - B 9 4 9 - C D E 0 0 F 2 1 9 7 2 A
�P kfrmID    �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 1 B 2 D C E E 6 - 6 4 9 5 - 4 2 8 8 - A 6 5 6 - 0 9 8 7 7 9 5 9 D C 0 D
�L kfrmID    �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H D 1 D 5 6 B 4 8 - F 1 A 3 - 4 A 5 B - 9 F A 1 - 9 A 4 A E A 9 4 2 D 6 B
�H kfrmID   "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H 4 7 5 B 8 4 2 5 - E 0 C D - 4 1 8 4 - 9 6 0 F - 8 7 0 2 5 C A 8 B 6 5 9
�D kfrmID   (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H D A 0 D 6 B 8 E - 5 9 0 C - 4 8 2 A - B 8 F 7 - 9 0 0 6 F 4 A 7 E 1 1 6
�@ kfrmID   .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H 4 7 0 5 D 0 D F - 7 D 6 0 - 4 8 0 7 - 8 A 8 4 - F 9 E 8 F 0 E 3 D 6 C 1
�< kfrmID   44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H A 5 6 9 2 A 9 C - 2 6 E 7 - 4 2 9 7 - B E F 3 - 8 D 7 6 9 7 F 2 0 1 C 3
�8 kfrmID   :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H 0 0 D A 4 2 8 5 - 7 6 E 9 - 4 1 0 1 - 9 D 1 F - F E 1 9 9 A 2 F D 5 F 9
�4 kfrmID   @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H B 8 9 E 4 7 1 C - 0 6 2 7 - 4 5 2 1 - 8 F B 7 - 0 C 9 F E 4 0 A C 1 1 A
�0 kfrmID   FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H 3 7 8 6 5 9 0 5 - C D 2 D - 4 D 6 5 - 9 7 2 A - 8 3 0 9 D E B B B A A 6
�, kfrmID   LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H 6 7 F 9 6 F 9 4 - D 1 C D - 4 E 7 C - 9 0 8 6 - D 1 2 A C C 3 2 C A B B
�( kfrmID   RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H F 5 1 3 0 7 1 C - 2 8 A 9 - 4 0 8 6 - B D F 9 - A 2 E D 1 6 0 4 3 B 7 2
�$ kfrmID   XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H 4 C 4 1 5 2 0 C - 2 3 7 0 - 4 8 5 4 - 9 4 3 D - 4 2 E D D 9 0 E 4 8 A E
�  kfrmID   ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H F A 7 B D 3 4 F - 9 5 7 7 - 4 6 7 E - 8 B F 1 - 4 3 F 5 5 E 5 4 1 1 8 6
� kfrmID   dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H D 6 A 8 E 5 C 1 - C 9 E 5 - 4 A B 4 - A C 7 3 - 8 5 1 1 6 B E 3 C E 6 0
� kfrmID   jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H E 2 9 6 C 9 7 7 - B 5 5 0 - 4 4 1 5 - B F 5 8 - 3 2 9 B 4 A 9 2 C 7 4 A
� kfrmID   pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H 0 B 5 8 4 6 7 B - C 6 1 D - 4 D 0 5 - 9 6 2 E - 2 2 7 2 C C 6 C 4 B A 1
� kfrmID    vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H 6 7 0 C 5 8 F E - 4 F F 3 - 4 C 0 E - A 7 4 C - 1 C 4 A E 6 4 7 E 1 D D
� kfrmID  ! || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H 9 F E 4 D E 2 A - 8 5 D D - 4 F 4 7 - B A 8 E - 6 0 A 1 2 F 1 D 2 6 D 2
� kfrmID  " �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 6 C F A 2 2 2 - C 1 2 5 - 4 5 C 2 - 9 6 A 6 - 1 0 1 8 9 4 6 9 2 A 4 E
� kfrmID  # �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H 1 2 7 2 5 7 F 7 - B C 4 B - 4 E 4 1 - 8 A 4 1 - E E 4 9 1 E 2 2 7 4 7 7
�  kfrmID  $ �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H F 7 4 F E 3 D 0 - 8 9 6 7 - 4 0 F D - 9 3 4 B - 7 A 9 E 1 7 6 0 5 9 6 4
�� kfrmID  % �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 F 6 E A C C D - 5 C 2 8 - 4 7 7 0 - B B B 9 - A 4 E 4 9 7 E 0 8 6 8 5
�� kfrmID  & �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 7 D C 8 6 2 8 - 3 F D 0 - 4 4 E E - B E 8 9 - F A D 1 6 A 5 A 5 A C F
�� kfrmID  ' �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E D 2 D 0 2 6 9 - 0 D D 3 - 4 D B 5 - 9 A 3 F - 5 A F 8 7 A 4 6 8 6 0 8
�� kfrmID  ( �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A E F 3 7 6 7 A - D 9 B E - 4 1 6 D - A C 7 4 - 7 A D F C 3 5 B A B D 6
�� kfrmID  ) �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 6 2 5 3 0 6 B - D 8 B 7 - 4 B 9 C - 9 C C 4 - C A 2 6 9 A 9 2 A B 5 3
�� kfrmID  * �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 9 2 4 E E 9 A 0 - 2 6 A D - 4 6 5 5 - 9 2 C D - 2 D 7 9 6 D D 6 F C B 0
�� kfrmID  + �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B 8 9 C 0 C 8 5 - B 5 7 9 - 4 2 3 E - B 4 7 7 - 8 B 8 4 E 0 2 1 4 8 B B
�� kfrmID  , �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 4 5 3 F 9 8 5 - 4 5 E E - 4 7 F 6 - 8 4 D F - 9 1 6 8 7 2 5 C 7 A 2 D
�� kfrmID  - �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 5 2 2 2 0 C D - 4 1 F 7 - 4 6 C D - B D C 7 - F 5 4 F 9 F E E 8 F 5 F
�� kfrmID  . �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 5 9 C 1 B 2 D - D 5 9 A - 4 5 C 8 - B 5 D A - 2 F 8 5 5 6 5 8 D 8 E 8
�� kfrmID  / �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H A 3 6 D 3 7 D B - D 8 A B - 4 A 2 B - A F 9 9 - D C 0 E B E 2 8 5 9 7 A
�� kfrmID  0 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 5 6 5 4 D 9 A - 6 4 1 E - 4 C 9 B - 9 6 A B - 0 D 5 0 D 4 9 A C 5 2 2
�� kfrmID  1 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 0 0 5 F C B 2 5 - 7 0 5 D - 4 0 F 4 - A D F 4 - F 1 B 4 B 5 1 F 3 5 4 4
�� kfrmID  2 �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E F D A 0 C 1 E - 9 D E 0 - 4 2 D A - 9 6 5 3 - 5 E 9 0 4 4 4 B 3 F 3 9
�� kfrmID  3 �� ������� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev� ��� H B 6 0 0 E E A 2 - 8 3 B 7 - 4 E F 5 - B F 5 A - 1 0 5 C 9 8 F D D 9 B 9
�� kfrmID  4 �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H E C D 9 C 1 2 6 - 7 A 7 4 - 4 1 1 3 - 8 C 0 F - D 5 1 A 1 2 0 A C 9 E F
� kfrmID  5 �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 E 2 A 5 C C F - 1 5 3 B - 4 4 C 5 - 8 1 8 1 - 0 B B 1 B 2 A F 5 5 3 6
� kfrmID  6 �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 3 A C 3 0 D A 5 - 2 4 1 6 - 4 F E 3 - A 8 2 D - 0 5 B 7 3 5 1 D A 9 D 7
� kfrmID  7    �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 1 3 8 1 C 1 B D - D C 5 F - 4 1 A 5 - A 8 A 4 - E 5 1 7 9 4 5 6 3 C E 2
� kfrmID  8  �� b�	�
� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 0 0 C F 6 2 E 3 - 2 3 6 0 - 4 2 1 0 - 8 E 6 3 - A 4 E 4 9 0 0 7 6 D B 8
� kfrmID  9  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 9 1 9 1 2 9 A 7 - 0 3 B 5 - 4 9 2 4 - A 3 D A - B 9 1 7 E B A 0 F C 1 D
� kfrmID  :  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 9 C 4 E E 8 D F - B 2 F E - 4 8 F F - 9 0 A 1 - 5 A D A D 4 9 7 1 3 C 7
� kfrmID  ;  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 0 9 B 4 3 A 0 6 - A 8 0 C - 4 D 1 4 - A D 4 A - A 9 8 9 3 D F 7 8 A A A
� kfrmID  <  � � b�!�
� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev  �## H 2 B B B 4 F 6 4 - 4 6 7 4 - 4 8 F D - 8 8 5 C - 1 D C 7 B 0 1 2 F 4 4 F
� kfrmID  = $$ %�&�% b�'�
� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev& �)) H 4 4 1 2 5 A 4 7 - D A 2 3 - 4 9 4 D - 9 3 1 7 - 0 A A C 5 1 D 1 0 E C 5
� kfrmID  > ** +�,�+ b�-�
� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev, �// H F 6 D A 4 A 3 D - 3 B 4 8 - 4 8 8 4 - 9 A 2 F - 6 C 2 0 8 B 0 F E 6 0 A
� kfrmID  ? 00 1�2�1 b��3��
�� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev2 �55 H 4 B 1 7 E 0 8 5 - A E 7 4 - 4 E 7 5 - 8 C 8 C - 3 2 1 7 D 5 D 3 E 5 8 6
� kfrmID  @ 66 7��8��7 b��9��
�� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev8 �;; H 4 D C B D B E 4 - 7 C 2 E - 4 4 8 F - 8 2 A C - E C F 4 8 5 F 2 4 9 5 F
�� kfrmID  A << =��>��= b��?��
�� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev> �AA H E 4 7 0 5 7 7 9 - E C 5 3 - 4 6 5 3 - 8 7 8 B - 7 C 8 5 2 D B 9 2 5 A 5
�� kfrmID  B BB C��D��C b��E��
�� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevD �GG H F 4 F 0 8 3 2 3 - C A 2 A - 4 D 0 8 - B 9 0 C - B B D E 1 9 E F 0 8 7 D
�� kfrmID  C HH I��J��I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
�� 
wrevJ �MM H E 9 7 D 9 8 A 3 - 5 E 0 E - 4 4 3 1 - A 6 B 7 - C 0 2 2 E A D 5 C 6 6 2
�� kfrmID  D NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H B 8 6 9 E 2 0 5 - 5 C A C - 4 A 9 3 - 9 4 F 1 - 0 A 2 3 5 5 1 A 7 D 9 9
�| kfrmID  E TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H 6 1 6 E 4 2 A 0 - A 0 2 A - 4 7 8 E - 9 8 D B - D 8 6 0 9 4 4 B 8 D 9 D
�x kfrmID  F ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H 7 8 A 3 4 B D B - A F 1 6 - 4 F F F - 8 D 2 A - F 3 2 1 B 0 3 8 F F 0 5
�t kfrmID  G `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H A F 8 C C D 0 B - 9 0 F 8 - 4 8 D D - 8 1 4 5 - C 7 C 0 C C A F F E 3 8
�p kfrmID  H ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H 3 E 1 5 A 8 9 4 - 8 5 A 7 - 4 8 E 1 - 9 F 7 3 - 8 7 D 1 2 E 8 E 3 D 2 7
�l kfrmID  I ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H 5 A C C E A 6 3 - 9 A E 3 - 4 9 4 A - B D 7 C - 5 9 1 C F C C E C A 8 C
�h kfrmID  J rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H 9 6 6 2 9 A C 7 - 1 4 F 3 - 4 D 2 E - 9 F 8 A - C C A 7 E 3 4 7 D 6 1 7
�d kfrmID  K xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H B A 1 9 4 D 4 1 - 1 F 4 D - 4 5 E 2 - A 8 6 E - 7 C 3 D F 1 8 4 D A 7 6
�` kfrmID  L ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 4 7 B D C 0 E 2 - B B 6 6 - 4 1 A 7 - 9 C 9 9 - D 4 C 3 5 D 7 6 8 9 0 9
�\ kfrmID  M �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H 4 7 8 A 6 7 E 9 - 8 A C A - 4 8 2 2 - A B 4 6 - F F 0 4 2 0 1 5 5 8 5 6
�X kfrmID  N �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H F 2 8 6 F F 8 5 - 5 1 7 1 - 4 4 4 1 - B 8 5 7 - B B 8 F E 2 A 0 8 7 B 3
�T kfrmID  O �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H A 9 B C 6 5 9 8 - 2 C E F - 4 6 E 4 - B C 2 2 - 4 5 C 7 E 9 7 E 3 0 9 2
�P kfrmID  P �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H 6 A 6 D D 9 1 5 - E 1 7 6 - 4 8 3 3 - 9 C 4 F - 5 2 2 D 4 9 5 1 B F 9 F
�L kfrmID  Q �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H D B 3 4 7 8 B 4 - 0 6 5 A - 4 8 6 D - A 9 9 8 - 7 5 1 4 1 1 F B 1 2 8 6
�H kfrmID  R �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H 1 D D 3 8 4 4 E - E 1 E C - 4 F 6 6 - 9 1 7 5 - 3 F 7 A 0 3 8 7 0 C 1 9
�D kfrmID  S �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H A 0 7 4 1 E 8 D - 9 D A 2 - 4 F E 2 - A F 4 F - E F A 2 C 5 7 4 C 9 9 F
�@ kfrmID  T �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H B B E 5 8 E 4 E - 3 2 5 9 - 4 A E 1 - B 1 B F - A 4 3 9 5 7 7 0 9 8 0 A
�< kfrmID  U �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H D E 1 3 7 0 B 4 - 3 9 6 B - 4 3 3 8 - A D 2 0 - A 7 4 F D 2 1 6 B D A 9
�8 kfrmID  V �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H 4 7 5 1 A 6 4 3 - 1 4 C 4 - 4 A 8 9 - 8 0 0 9 - 5 D 4 C 7 8 0 B D C C 2
�4 kfrmID  W �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H B B 5 4 2 E A 6 - 8 1 8 C - 4 E 0 6 - 8 1 2 8 - D 6 5 3 B A 7 F C 1 6 0
�0 kfrmID  X �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H D A E 5 E 8 8 B - 1 F A 4 - 4 6 5 D - 8 4 2 8 - F 0 0 D E 4 C 1 B 8 F D
�, kfrmID  Y �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H 6 D 6 F A D 4 F - 9 E B 9 - 4 E E F - B 1 6 4 - 3 E 8 6 E 5 4 C 1 D D 4
�( kfrmID  Z �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H 5 6 5 5 0 2 8 9 - E 7 5 4 - 4 2 7 B - 8 9 2 9 - 8 4 E 2 9 6 4 F 6 0 5 4
�$ kfrmID  [ �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H 9 E D 7 D 1 1 1 - 7 4 0 E - 4 A E 6 - A 6 8 A - 1 5 6 4 8 4 2 9 2 C 5 0
�  kfrmID  \ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 8 1 6 B B 7 C F - 7 6 6 0 - 4 D 3 0 - B 1 A C - 4 1 E 7 8 1 9 7 9 9 7 2
� kfrmID  ] �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H B 5 0 2 4 A 7 4 - F 2 9 E - 4 7 8 7 - A 0 E A - 8 7 B 9 D 9 C A 5 D 7 9
� kfrmID  ^ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 1 8 C B 6 4 7 - E 8 1 C - 4 C 0 3 - 9 0 D 5 - A 1 8 6 F 5 D 0 E A E 2
� kfrmID  _ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 1 0 E 7 2 E 0 B - 6 1 9 6 - 4 9 4 B - 8 7 7 0 - C 3 B 1 8 F 7 B 8 2 3 A
� kfrmID  ` �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H 8 D 8 A C E F C - 1 4 2 C - 4 3 1 E - B D F 3 - 3 7 7 A 6 2 6 A B D C 4
� kfrmID  a �� ��	��� b���
� 
wres� �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� � H E 2 7 0 E E 3 B - B E 7 2 - 4 6 1 9 - 9 C 6 7 - 6 4 D 5 D 9 C A 3 A F 8
� kfrmID  b  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 4 7 9 A 2 5 8 4 - 5 7 C A - 4 B E 2 - 9 B 9 D - 1 E D 7 2 0 6 9 1 5 5 4
� kfrmID  c  	�
� 	 b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev
 � H 4 6 1 A 6 5 4 3 - E D B B - 4 9 8 E - A 1 C 2 - 4 8 6 0 F C E E 1 8 5 5
�  kfrmID  d  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H D F D 8 4 2 2 F - 9 A 1 8 - 4 2 6 B - B A A 2 - 0 1 D 0 D A 2 6 6 2 E 9
�� kfrmID  e  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H D 7 4 9 4 D 8 E - E B A 1 - 4 D E 8 - B B 8 B - C 2 8 2 5 5 3 C 9 5 8 4
�� kfrmID  f  ���� b����
�� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � H 2 1 F A 0 6 3 8 - E F 8 C - 4 1 F 9 - B D 2 9 - C 5 0 C A 8 9 D 1 8 0 3
�� kfrmID  g    !��"��! b��#��
�� 
wres# �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev" �%% H F C 6 8 D F 4 F - 0 5 6 6 - 4 1 A 0 - 8 6 3 E - F B 6 7 F F 1 A 7 2 C 4
�� kfrmID  h && '��(��' b��)��
�� 
wres) �** H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev( �++ H E 2 7 3 3 2 4 B - C F 1 9 - 4 8 F 3 - A 2 4 9 - 6 4 3 6 C B A 2 C 3 A 2
�� kfrmID  i ,, -��.��- b��/��
�� 
wres/ �00 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev. �11 H 6 6 3 C B 0 8 D - B 2 9 F - 4 6 5 9 - A D 5 5 - F 1 B A B 7 C 2 0 7 4 F
�� kfrmID  j 22 3��4��3 b��5��
�� 
wres5 �66 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev4 �77 H B C 5 8 6 5 1 2 - 5 5 7 7 - 4 7 C F - 9 5 C A - 9 E 7 0 7 E 4 C 1 A 2 8
�� kfrmID  k 88 9��:��9 b��;��
�� 
wres; �<< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev: �== H 7 9 B 4 C 6 D F - F 1 C 3 - 4 F 9 B - 8 2 3 4 - 8 A 3 6 0 8 0 9 B A D D
�� kfrmID  l >> ?��@��? b��A��
�� 
wresA �BB H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev@ �CC H E 0 C 1 A 1 C 3 - 2 F B 5 - 4 1 3 A - B B 9 D - 1 4 0 A 6 0 A 3 4 4 A F
�� kfrmID  m DD E��F��E b��G��
�� 
wresG �HH H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevF �II H 9 2 8 8 3 D D 9 - 2 F F 5 - 4 E 5 4 - 9 9 E 0 - 6 6 3 C 8 4 3 3 7 8 7 C
�� kfrmID  n JJ K��L��K b��M��
�� 
wresM �NN H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevL �OO H 1 B 5 F 3 B 8 5 - 4 B F 5 - 4 1 C A - A 2 4 C - A F 0 D E 7 0 8 2 4 7 B
�� kfrmID  o PP Q��R��Q b��S��
�� 
wresS �TT H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevR �UU H A 9 C D B 1 C B - 5 4 1 0 - 4 1 7 8 - B 3 C B - E 5 7 2 1 E 8 7 0 1 B 5
�� kfrmID  p VV W��X��W b��Y��
�� 
wresY �ZZ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevX �[[ H E 7 4 2 F C B 9 - 6 5 3 7 - 4 6 E A - B 2 4 3 - F 1 E 2 9 C F B 0 D 1 4
�� kfrmID  q \\ ]��^��] b��_��
�� 
wres_ �`` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev^ �aa H B F 0 2 B A 9 B - B E 5 E - 4 7 7 E - 9 2 5 3 - 8 3 6 A 1 5 5 7 8 F 7 7
�� kfrmID  r bb c��d��c b��e��
�� 
wrese �ff H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrevd �gg H 1 9 0 7 9 4 9 D - 6 D 4 6 - 4 9 3 1 - 8 1 9 B - 4 7 1 4 1 3 2 F 7 6 2 1
�� kfrmID  s hh i��j��i b�k�
� 
wresk �ll H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrevj �mm H E E 9 9 2 D D 0 - 8 7 E 3 - 4 F 1 3 - B 7 9 2 - D F 6 0 4 7 6 9 B F 0 3
�� kfrmID  t nn o�p�o b�q�
� 
wresq �rr H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevp �ss H 3 9 8 C 6 8 2 E - B 1 8 C - 4 7 9 A - A A A 5 - A 4 1 D 5 D 2 B E 4 E F
� kfrmID  u tt u�v�u b�w�
� 
wresw �xx H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevv �yy H 3 B D C 1 4 4 C - 3 E 4 2 - 4 0 5 C - 9 5 E E - 3 E 0 5 6 C 3 1 2 6 D 5
� kfrmID  v zz {�|�{ b�}�
� 
wres} �~~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev| � H D 7 5 A 3 C 4 C - 0 D C 0 - 4 5 9 A - A F 7 A - F A 1 3 F E 8 2 E D 7 8
� kfrmID  w �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 9 F 7 0 B 3 4 9 - E D 7 3 - 4 9 8 6 - 8 4 B 6 - 9 3 F F 5 3 3 D 2 E B C
� kfrmID  x �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H B 2 2 0 0 4 3 7 - F E 1 9 - 4 4 3 B - 8 3 B B - 2 C 6 B 4 C 2 F B C C 1
� kfrmID  y �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 7 0 9 3 0 5 F 3 - 9 A 8 7 - 4 2 1 8 - 8 0 7 9 - 7 7 F B 1 6 D A D 4 5 D
� kfrmID  z �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 6 8 C 3 5 D 9 7 - 8 9 6 A - 4 9 4 5 - 9 4 A C - 4 B 3 2 1 E E D 1 B F C
� kfrmID  { �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 2 B 5 6 0 3 1 - 7 9 3 3 - 4 8 9 1 - 9 F B C - 1 1 F F D E A 9 7 7 2 F
� kfrmID  | �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 8 F A 0 1 0 7 F - 9 2 F 0 - 4 4 B 4 - B 5 F 1 - B F E E 6 7 7 7 7 A 5 1
� kfrmID  } �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 8 5 A A E 1 8 D - D C 4 B - 4 3 0 2 - B 9 E E - F 9 D C 2 5 1 F 5 F F A
� kfrmID  ~ �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 4 8 4 2 3 0 8 C - 7 2 1 6 - 4 4 8 2 - 8 5 2 F - C D C 7 B 2 8 C 7 2 E B
� kfrmID   �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 8 B D 3 6 8 8 - 8 C 0 A - 4 8 F C - 9 C 2 F - E A 3 5 A F 7 F 7 6 1 2
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 1 1 8 B D 2 6 - 6 2 3 E - 4 8 6 E - 8 F F D - 6 8 9 8 F D E B 0 7 C B
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H C 8 D A E 0 A 5 - 6 1 9 6 - 4 8 2 9 - 9 8 6 2 - 5 E 8 5 6 0 D 1 4 7 B C
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 C 3 E 6 8 B 8 - 6 C E 1 - 4 4 A 9 - 8 E 3 E - 6 E 5 2 6 1 B 6 2 C F B
� kfrmID  � �� ����� b���~
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev� ��� H 9 4 4 9 A 3 3 7 - 6 4 5 A - 4 6 3 5 - 8 A 2 6 - F 5 C 2 4 1 C 2 2 6 F E
� kfrmID  � �� ��}��|� b�{��z
�{ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev� ��� H E F D 3 2 A 6 6 - 1 6 C 9 - 4 6 C 2 - A 0 4 E - E E F 1 7 8 2 5 E 7 1 A
�| kfrmID  � �� ��y��x� b�w��v
�w 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev� ��� H D F 8 3 2 7 7 9 - D 0 C 9 - 4 D 3 1 - A 8 0 6 - 0 4 8 5 2 6 5 2 0 D A A
�x kfrmID  � �� ��u��t� b�s��r
�s 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev� ��� H A 9 4 6 4 A 0 F - A 1 3 3 - 4 6 B 3 - A 1 3 1 - B 6 9 3 4 D 6 5 8 4 9 0
�t kfrmID  � �� ��q��p� b�o��n
�o 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev� ��� H 6 5 5 4 C 1 4 B - F 6 9 6 - 4 1 D 6 - 9 E C 8 - D 4 0 9 A 7 1 5 9 0 1 9
�p kfrmID  � �� ��m��l� b�k��j
�k 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev� ��� H 1 6 C 4 B B 2 9 - 0 6 B A - 4 2 8 0 - 8 8 7 D - 1 8 F B F B D 3 0 3 E 5
�l kfrmID  � �� ��i��h� b�g��f
�g 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev� ��� H D F E E 0 9 D C - 2 7 4 0 - 4 F A A - 9 9 D 2 - 1 5 D E F 8 0 B D 7 2 6
�h kfrmID  � �� ��e��d� b�c��b
�c 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev� ��� H C 8 2 8 1 F 6 C - A 3 7 9 - 4 E A F - B F 4 3 - E 3 1 F F 1 E 5 C 4 E D
�d kfrmID  � �� ��a��`� b�_��^
�_ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev� ��� H 2 4 C E 1 F C D - 8 6 1 5 - 4 D 6 3 - B E 4 0 - 6 B D 5 6 3 1 0 0 7 1 2
�` kfrmID  � �� ��] �\� b�[�Z
�[ 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev  � H 0 5 2 7 A D 3 0 - 3 8 E F - 4 B 2 2 - B 0 F 4 - 2 F A F 4 A C F 5 8 7 6
�\ kfrmID  �  �Y�X b�W�V
�W 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev �		 H 6 8 0 F A F D 3 - 0 B 6 7 - 4 7 2 6 - B 7 C A - 7 E 3 9 3 0 4 4 E 4 1 B
�X kfrmID  � 

 �U�T b�S�R
�S 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev � H 8 B 7 9 D A 2 5 - A 9 3 7 - 4 C 9 E - 8 C 5 E - 1 9 3 5 F E 3 5 9 F 9 9
�T kfrmID  �  �Q�P b�O�N
�O 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev � H 5 A 5 9 C 9 8 C - F 8 E 8 - 4 E 6 8 - B 1 1 5 - 3 7 8 E C 3 2 D B 5 4 9
�P kfrmID  �  �M�L b�K�J
�K 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev � H 0 E 8 A D 8 D F - 8 5 A 5 - 4 B 6 3 - A 2 4 7 - 8 7 D F 1 5 1 7 3 2 B 0
�L kfrmID  �  �I�H b�G�F
�G 
wres �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev �!! H 9 9 2 E 1 6 5 4 - B E A 0 - 4 E 4 8 - A E D 0 - C 5 F 6 5 B 4 F 2 4 E 0
�H kfrmID  � "" #�E$�D# b�C%�B
�C 
wres% �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$ �'' H 4 2 D D 4 6 3 B - 4 7 F 4 - 4 E B 2 - 9 F E D - E 0 3 2 F 9 6 F 6 4 F 9
�D kfrmID  � (( )�A*�@) b�?+�>
�? 
wres+ �,, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev* �-- H D 7 8 6 B 3 5 0 - 8 4 A 1 - 4 B D 3 - 9 5 B D - 9 D 1 E 0 F D 9 C 7 B 5
�@ kfrmID  � .. /�=0�</ b�;1�:
�; 
wres1 �22 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev0 �33 H 7 8 F 8 3 A 1 4 - C 9 0 8 - 4 2 E B - 9 7 3 C - A 9 3 B 3 8 1 4 4 5 4 0
�< kfrmID  � 44 5�96�85 b�77�6
�7 
wres7 �88 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev6 �99 H 6 9 9 D 3 3 F 3 - 9 4 F E - 4 2 0 4 - 8 6 3 D - 2 1 4 0 1 C E F E F 4 6
�8 kfrmID  � :: ;�5<�4; b�3=�2
�3 
wres= �>> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev< �?? H 6 5 9 4 1 A 0 B - 6 7 1 9 - 4 B C B - B 5 D B - B 7 E D 0 7 3 B C D 9 B
�4 kfrmID  � @@ A�1B�0A b�/C�.
�/ 
wresC �DD H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrevB �EE H 9 C 0 F 5 6 3 A - A 2 0 0 - 4 7 8 3 - 8 6 1 3 - 3 7 E A B 7 B 4 B 4 5 6
�0 kfrmID  � FF G�-H�,G b�+I�*
�+ 
wresI �JJ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrevH �KK H A E 5 6 8 2 0 A - E B E 0 - 4 E 1 E - A 3 D 7 - 5 6 5 1 3 7 3 3 2 5 9 5
�, kfrmID  � LL M�)N�(M b�'O�&
�' 
wresO �PP H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrevN �QQ H 0 F 8 E E A 3 2 - 3 1 4 5 - 4 B E 8 - B 4 D 6 - B A C 3 B 1 1 6 C 3 1 0
�( kfrmID  � RR S�%T�$S b�#U�"
�# 
wresU �VV H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrevT �WW H B 2 4 0 C 6 A F - 2 E B 9 - 4 A 9 C - A D B 1 - 3 1 9 D A 9 5 D 1 E A 1
�$ kfrmID  � XX Y�!Z� Y b�[�
� 
wres[ �\\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrevZ �]] H 9 7 5 8 8 E D 2 - E 4 F A - 4 1 2 4 - 8 0 8 2 - 4 A 9 D 8 2 4 8 6 7 C 3
�  kfrmID  � ^^ _�`�_ b�a�
� 
wresa �bb H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev` �cc H B 1 D E 4 F 7 3 - 6 7 2 A - 4 D 3 8 - 9 8 0 A - F 9 7 4 8 0 F F A 9 5 5
� kfrmID  � dd e�f�e b�g�
� 
wresg �hh H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevf �ii H F E 6 5 6 F 5 B - 5 1 0 9 - 4 9 2 A - 9 E B D - 9 7 3 4 A 6 3 8 0 3 6 4
� kfrmID  � jj k�l�k b�m�
� 
wresm �nn H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevl �oo H 4 7 E 4 A 4 3 A - 2 0 E 0 - 4 E 6 C - 9 8 1 1 - 5 1 3 2 5 4 E C 6 0 5 7
� kfrmID  � pp q�r�q b�s�
� 
wress �tt H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevr �uu H 6 0 B 9 0 1 0 5 - 5 5 E 0 - 4 4 B B - A 3 C 5 - 8 0 B B 3 6 4 A 7 4 E D
� kfrmID  � vv w�x�w b�y�

� 
wresy �zz H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrevx �{{ H 6 F F E 2 1 B E - 0 0 3 7 - 4 6 0 B - B A E 0 - F 1 6 3 8 2 4 7 C E 9 7
� kfrmID  � || }�	~�} b��
� 
wres ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev~ ��� H 7 1 C A D 2 0 E - 9 3 3 D - 4 0 E 1 - 9 9 A F - 8 E 2 1 5 F F 8 F 3 5 A
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 D 9 0 9 5 8 E - E 2 A B - 4 B D 4 - 8 5 1 2 - 9 0 8 5 9 5 0 D 8 0 4 1
� kfrmID  � �� ���� � b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev� ��� H B B E C F C 1 0 - D 6 3 4 - 4 D 5 6 - 9 3 6 0 - D 0 8 2 8 A C 0 7 8 6 6
�  kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 9 3 9 1 2 F D 0 - 5 0 7 B - 4 9 A 5 - B F 7 9 - 6 6 6 C 1 8 2 F D 6 5 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H B B A 8 9 2 E 3 - 3 E E A - 4 1 1 0 - B C 3 E - 5 2 6 F 5 F A 6 A 3 B 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H C 5 2 5 7 0 F 7 - 0 7 0 7 - 4 5 D A - 8 F F 5 - 0 1 B F E 1 1 5 A C F D
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 1 F 9 7 8 0 5 7 - 4 D 8 2 - 4 F 4 B - 9 A D 1 - 9 7 1 8 4 6 0 5 7 7 3 5
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 D 3 F 7 F A 6 - A 8 8 3 - 4 0 8 E - 8 E 5 4 - 9 4 E 5 C 9 4 8 A C 3 3
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 2 C C 6 7 9 A 1 - D D A D - 4 4 C 1 - 9 0 F C - C E A B A F 2 9 B B 8 3
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H D B 9 7 9 B A E - 0 4 9 7 - 4 C D 3 - 9 4 F 1 - B 8 C 7 B 1 2 8 8 D B 2
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 4 8 B 2 D 5 2 8 - 6 0 9 2 - 4 3 7 5 - 8 5 4 9 - 6 A A 3 A 8 C 3 B 4 4 8
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 5 9 6 5 7 3 D D - 0 5 F 5 - 4 9 8 1 - 9 2 0 4 - D 1 4 2 E 6 5 7 E 5 B 4
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 3 6 9 D 4 1 F - 3 8 B 0 - 4 2 0 8 - 8 8 F 5 - 0 0 1 8 7 4 1 6 8 1 3 9
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 8 B 7 4 9 4 1 F - B B 4 1 - 4 8 2 A - A 8 6 1 - D F 3 A E 2 9 4 0 A F 1
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 2 E 8 9 3 E 1 - E 8 9 1 - 4 5 2 8 - B 4 0 E - 3 8 A 5 3 8 9 3 A B 2 0
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 7 8 F A 5 D 5 2 - 3 C F E - 4 3 E 3 - A 7 2 5 - 9 2 B 2 5 4 1 6 D 6 0 2
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H 6 4 F 0 C 8 4 9 - 8 A 7 2 - 4 5 2 8 - 9 1 A 2 - B 7 3 8 2 2 A 8 4 2 5 E
�� kfrmID  � �� ������� b�����
�� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev� ��� H E 4 D D 0 3 1 2 - B 2 3 4 - 4 3 8 F - A 2 6 4 - 7 5 1 3 7 7 8 6 B 8 E 5
�� kfrmID  � �� ������� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev� ��� H 3 B 4 8 B 9 7 7 - D 7 C F - 4 3 C 7 - 9 2 A 8 - 2 B C 7 0 3 D 1 8 1 3 D
�� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H B 8 E 6 9 8 A 6 - 3 F C A - 4 B E 9 - B 3 9 9 - D B 9 5 5 E D C B 7 F 9
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H D 5 F F 8 B 7 0 - D 8 5 B - 4 9 7 8 - 8 3 3 E - 1 3 0 E 0 1 B 5 5 3 5 8
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 5 9 A F 6 7 7 E - C 0 C B - 4 7 D 9 - 9 B E 6 - 8 4 1 B 8 2 7 E E F E A
� kfrmID  �    �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H A F 7 9 0 1 0 1 - 9 1 2 6 - 4 E 3 6 - A 0 3 B - B 9 4 8 9 1 2 2 5 5 5 C
� kfrmID  �  �� b�	�
� 
wres	 �

 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 4 7 2 D 1 8 F 0 - 6 A 0 2 - 4 7 4 A - 9 6 5 B - 6 4 4 5 6 2 4 3 7 E F B
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H C B 5 5 0 1 1 3 - 3 0 3 9 - 4 C 0 8 - 8 2 6 A - 4 B B 3 E 8 6 4 5 E B 3
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 5 6 B F E 4 9 7 - 3 5 2 2 - 4 4 A 0 - 8 2 F E - 1 F E E F 7 3 C 6 3 D 4
� kfrmID  �  �� b��
� 
wres � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � H 1 F 3 B 9 6 7 F - D F F E - 4 E 5 2 - 8 A 3 C - 8 E A B 8 A A 5 0 8 4 4
� kfrmID  �  � � b�!�
� 
wres! �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev  �## H 1 8 F D 6 8 0 E - C 3 5 9 - 4 2 E 9 - A 1 7 E - 8 F B 0 D 5 A 2 B 9 B 5
� kfrmID  � $$ %�&�% b�'�
� 
wres' �(( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev& �)) H F 5 0 6 E 5 5 3 - 7 1 8 6 - 4 7 2 5 - 9 7 D D - C 6 4 3 E 1 3 F 9 2 7 F
� kfrmID  � ** +�,�+ b�-�
� 
wres- �.. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev, �// H 2 7 1 0 D F 1 2 - C 2 A 5 - 4 4 4 3 - A A 0 9 - 9 7 4 E A 4 E B E 3 C E
� kfrmID  � 00 1�2�1 b�3�
� 
wres3 �44 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev2 �55 H 2 F 1 9 2 4 8 1 - 6 7 C B - 4 4 0 A - 9 4 D 7 - 9 C B A 3 8 D E 8 7 4 6
� kfrmID  � 66 7�8�7 b�9�
� 
wres9 �:: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev8 �;; H 0 A B 7 F A B A - 6 B D 8 - 4 B 6 1 - 8 7 5 9 - 9 5 4 F 2 9 5 B E B F B
� kfrmID  � << =�>�= b�?�
� 
wres? �@@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev> �AA H 2 4 2 1 F 2 C 7 - 4 4 5 D - 4 3 2 3 - 8 D D 2 - 0 1 2 5 5 F 2 A 8 F E 7
� kfrmID  � BB C�D�C b�E�
� 
wresE �FF H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrevD �GG H 2 F 9 7 D 0 E 2 - A 9 7 C - 4 D 0 D - 9 7 2 7 - 5 F 4 4 8 B 1 A 7 6 A 4
� kfrmID  � HH I�J�I b�K�~
� 
wresK �LL H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrevJ �MM H B B A C 7 5 7 A - 2 9 6 0 - 4 3 3 B - B E B C - 2 A D 4 7 C D 8 3 C 7 2
� kfrmID  � NN O�}P�|O b�{Q�z
�{ 
wresQ �RR H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrevP �SS H 8 1 5 C C 7 D D - 3 7 B 6 - 4 7 B 1 - B F E D - E 9 7 0 0 5 6 0 3 5 E A
�| kfrmID  � TT U�yV�xU b�wW�v
�w 
wresW �XX H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrevV �YY H C 0 8 E F 0 C 6 - 2 3 5 9 - 4 7 2 E - 8 8 6 8 - 7 B A 6 E C 1 2 F A 9 9
�x kfrmID  � ZZ [�u\�t[ b�s]�r
�s 
wres] �^^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev\ �__ H C 5 5 8 B D 4 E - 4 2 C 9 - 4 8 A 7 - B 0 1 3 - B 9 9 F E E 3 6 5 5 1 E
�t kfrmID  � `` a�qb�pa b�oc�n
�o 
wresc �dd H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrevb �ee H 2 A B 8 0 9 3 D - D A 6 0 - 4 7 A 7 - A E B 1 - D 0 C 1 3 6 8 6 4 4 2 8
�p kfrmID  � ff g�mh�lg b�ki�j
�k 
wresi �jj H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrevh �kk H 2 0 F C C D 0 2 - 2 1 9 3 - 4 B 1 1 - B 1 D A - B 0 D 8 3 B 9 B 2 A D 8
�l kfrmID  � ll m�in�hm b�go�f
�g 
wreso �pp H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrevn �qq H C 6 F E 0 8 1 B - 4 4 3 8 - 4 A 4 C - A 2 3 2 - D B 4 6 C 0 0 8 F 0 B E
�h kfrmID  � rr s�et�ds b�cu�b
�c 
wresu �vv H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrevt �ww H D E C 7 3 3 F 1 - 5 5 4 8 - 4 7 C 1 - B C 5 F - 5 1 4 A 7 9 D 4 7 A 1 7
�d kfrmID  � xx y�az�`y b�_{�^
�_ 
wres{ �|| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrevz �}} H B D F B 0 7 8 3 - 3 3 5 0 - 4 8 0 5 - 9 3 4 A - 7 5 3 5 1 C 4 D 1 5 A 4
�` kfrmID  � ~~ �]��\ b�[��Z
�[ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev� ��� H 9 0 C B 0 3 E E - 2 D 6 5 - 4 6 0 A - A B B 5 - A 5 0 2 3 F D 3 4 F 9 F
�\ kfrmID  � �� ��Y��X� b�W��V
�W 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev� ��� H A C 7 3 2 0 A 2 - C F 9 6 - 4 9 B E - 9 A 8 C - 1 0 1 9 9 5 5 2 3 5 4 4
�X kfrmID  � �� ��U��T� b�S��R
�S 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev� ��� H A E 5 5 5 F 3 1 - D 0 D C - 4 5 4 F - 8 2 4 1 - 7 8 6 8 6 D B 0 7 A 3 1
�T kfrmID  � �� ��Q��P� b�O��N
�O 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev� ��� H F 2 1 F 8 2 8 1 - 4 B 5 B - 4 1 3 4 - 9 4 B C - 7 1 A 4 0 B A 6 A F 2 D
�P kfrmID  � �� ��M��L� b�K��J
�K 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev� ��� H 8 F E 1 B 5 2 0 - 6 F 6 6 - 4 7 7 5 - B 9 5 2 - A A F 2 1 C 4 3 8 9 3 4
�L kfrmID  � �� ��I��H� b�G��F
�G 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev� ��� H 8 C 4 8 D 7 0 A - 5 2 8 E - 4 A 3 0 - A F C 0 - B 3 0 5 9 0 5 6 F A 6 F
�H kfrmID  � �� ��E��D� b�C��B
�C 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev� ��� H 7 9 3 A A F C A - C 4 3 2 - 4 B 2 0 - A 6 C D - 1 8 A 8 6 5 D D B 9 F 9
�D kfrmID  � �� ��A��@� b�?��>
�? 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev� ��� H 9 4 4 2 A E B 4 - 4 4 3 0 - 4 3 5 2 - A E 0 6 - 9 9 5 E F F 4 2 C 3 0 2
�@ kfrmID  � �� ��=��<� b�;��:
�; 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev� ��� H A A 5 F 4 A C 8 - D A 3 B - 4 8 4 0 - A 8 C 7 - 4 E 5 3 F 7 4 5 3 5 4 3
�< kfrmID  � �� ��9��8� b�7��6
�7 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev� ��� H F 2 D 0 7 5 5 6 - D E F E - 4 A 2 1 - A 4 5 6 - 8 E 3 F 3 4 5 F 9 7 9 A
�8 kfrmID  � �� ��5��4� b�3��2
�3 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev� ��� H 0 6 C 0 8 0 8 5 - C 1 4 C - 4 C 1 9 - 8 5 2 F - 4 C 2 1 6 B 0 0 9 B B 3
�4 kfrmID  � �� ��1��0� b�/��.
�/ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev� ��� H 4 8 0 2 A F 4 0 - 3 9 5 F - 4 0 C 9 - A 4 0 7 - 9 3 6 9 C 3 0 A 6 A F 8
�0 kfrmID  � �� ��-��,� b�+��*
�+ 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev� ��� H 2 A F 2 F 3 0 9 - 7 8 F 2 - 4 E F 7 - 8 6 E C - 1 7 C 1 2 5 4 2 0 1 7 3
�, kfrmID  � �� ��)��(� b�'��&
�' 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev� ��� H A 5 C A 3 8 8 3 - C C B C - 4 6 6 9 - 9 B 1 4 - 2 0 8 4 5 B 8 8 A 6 6 C
�( kfrmID  � �� ��%��$� b�#��"
�# 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev� ��� H B 6 8 2 3 4 9 3 - C 0 2 3 - 4 B 4 5 - 8 A D C - F 0 0 3 3 D 2 7 3 8 0 E
�$ kfrmID  � �� ��!�� � b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev� ��� H 0 5 4 8 B 7 D 2 - 3 6 E 3 - 4 3 1 7 - A 1 1 F - 3 E 3 7 E A 7 D 9 F C 7
�  kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H A 5 6 8 0 A 2 A - A 0 1 8 - 4 F 0 C - 9 A 0 C - F D 6 1 A 4 3 A F 2 A 4
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 3 7 E E 9 5 9 6 - 1 8 C 4 - 4 0 9 5 - B A E B - E A D D 7 E 3 D C 6 A 9
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 1 2 1 0 5 4 5 9 - 7 1 B 1 - 4 7 6 0 - A 9 B A - 8 B 9 8 9 9 2 1 9 2 F 4
� kfrmID  � �� ����� b���
� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev� ��� H 0 5 F 7 3 7 1 0 - 6 3 B 1 - 4 B 2 A - A 9 E 0 - E A E 6 6 F 3 7 F 2 A 5
� kfrmID  � �� ����� b���

� 
wres� ��� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev� ��� H 9 3 E 3 6 9 9 0 - 5 A 2 D - 4 6 5 0 - 8 5 E D - 0 A E B 8 D 1 B E 6 7 F
� kfrmID  � �� ��	��� b���
� 
wres� �     H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev� �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
� kfrmID  �     � �  b� �
� 
wres  �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev  �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
� kfrmID  �     	� 
�  	 b�� ��
�� 
wres  �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev 
 �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�  kfrmID  �     �� ��  b�� ��
�� 
wres  �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �     �� ��  b�� ��
�� 
wres  �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �     �� ��  b�� ��
�� 
wres  �   H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev  �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �       !�� "�� ! b�� #��
�� 
wres # � $ $ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev " � % % H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  & &  '�� (�� ' b�� )��
�� 
wres ) � * * H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev ( � + + H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  , ,  -�� .�� - b�� /��
�� 
wres / � 0 0 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev . � 1 1 H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  2 2  3�� 4�� 3 b�� 5��
�� 
wres 5 � 6 6 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev 4 � 7 7 H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  8 8  9�� :�� 9 b�� ;��
�� 
wres ; � < < H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev : � = = H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  > >  ?�� @�� ? b�� A��
�� 
wres A � B B H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev @ � C C H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  D D  E�� F�� E b�� G��
�� 
wres G � H H H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev F � I I H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  J J  K�� L�� K b�� M��
�� 
wres M � N N H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev L � O O H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  P P  Q�� R�� Q b�� S��
�� 
wres S � T T H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev R � U U H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  V V  W�� X�� W b�� Y��
�� 
wres Y � Z Z H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev X � [ [ H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  \ \  ]�� ^�� ] b�� _��
�� 
wres _ � ` ` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev ^ � a a H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  b b  c�� d�� c b�� e��
�� 
wres e � f f H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev d � g g H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  h h  i�� j�� i b�� k��
�� 
wres k � l l H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev j � m m H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  n n  o�� p�� o b�� q��
�� 
wres q � r r H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev p � s s H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  t t  u�� v�� u b�� w��
�� 
wres w � x x H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev v � y y H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  z z  {�� |�� { b�� }��
�� 
wres } � ~ ~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev | �   H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  � �  ��� ��� � b�� ���
�� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � � � � H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  � �  ��� ��� � b�� ���
�� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � � � � H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  � �  ��� ��� � b�� ���
�� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � � � � H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  � �  ��� ��� � b�� ���
�� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev � � � � H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  � �  ��� ��� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev � � � � H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
�� kfrmID  �  � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H A 0 C C C D 0 F - A C F 6 - 4 4 A 6 - 9 9 2 E - D C E E B 4 9 0 5 6 3 2
� kfrmID  �  � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H B 0 B B 0 7 C 3 - 4 7 9 9 - 4 8 D 9 - B 1 4 F - 0 8 2 9 1 D 0 9 9 8 4 E
� kfrmID  �  � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H B 0 B B 0 7 C 3 - 4 7 9 9 - 4 8 D 9 - B 1 4 F - 0 8 2 9 1 D 0 9 9 8 4 E
� kfrmID  �  � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H 3 D 4 3 9 8 2 5 - 0 4 1 E - 4 C 4 E - 9 F 7 D - A 6 6 2 8 F 0 8 D 6 E 1
� kfrmID     � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H C 7 D E 9 E D 9 - 0 E 4 3 - 4 3 9 1 - B 0 3 6 - B 0 E B C 0 E 2 3 9 9 E
� kfrmID    � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H D 3 7 C B B 1 5 - 1 C 7 3 - 4 6 7 E - B 9 C B - 2 8 A 7 7 2 7 5 6 E A 0
� kfrmID    � �  �� �� � b� ��
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev � � � � H 9 C 2 3 0 E 3 D - 6 4 9 7 - 4 2 4 0 - 8 8 0 C - 8 1 4 F F 1 0 D C F 5 8
� kfrmID    � �  �� �� � b� ��~
� 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev � � � � H 4 1 C A A 3 A 7 - 0 2 2 0 - 4 6 F 3 - A 3 B 4 - 5 9 1 6 C A E 3 B 2 4 4
� kfrmID    � �  ��} ��| � b�{ ��z
�{ 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev � � � � H 0 0 F F C 1 7 F - 8 8 A 2 - 4 3 3 F - A C 5 1 - E 5 5 6 8 1 C 5 6 B 5 0
�| kfrmID    � �  ��y ��x � b�w ��v
�w 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev � � � � H 4 0 6 0 A 6 9 6 - 6 0 B 5 - 4 8 2 1 - 9 E 3 1 - 2 0 2 E 7 A F E 9 3 2 F
�x kfrmID    � �  ��u ��t � b�s ��r
�s 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev � � � � H 8 0 F E 9 3 8 7 - 4 8 1 B - 4 4 2 1 - A E E B - 0 5 D 3 F 6 8 A 9 3 A D
�t kfrmID    � �  ��q ��p � b�o ��n
�o 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev � � � � H 8 8 5 8 2 7 6 9 - A 8 8 D - 4 7 F 4 - 8 D B B - C 1 C B 1 B 5 5 0 C 3 6
�p kfrmID    � �  ��m ��l � b�k ��j
�k 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev � � � � H 0 4 3 E E D F 2 - 8 3 5 6 - 4 7 E 7 - 9 B 1 5 - 7 4 A 7 3 3 4 3 B 2 D 5
�l kfrmID  	  � �  ��i ��h � b�g ��f
�g 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev � � � � H 8 E 1 A 3 1 1 B - B E 4 1 - 4 D C 7 - 8 2 6 5 - 9 3 A 8 9 F 4 D E F 7 0
�h kfrmID  
  � �  ��e ��d � b�c ��b
�c 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev � � � � H F 9 8 7 1 D 6 2 - 6 3 7 B - 4 9 E D - A 0 7 4 - A F 4 7 2 8 0 B 7 D A 4
�d kfrmID    � �  ��a ��` � b�_ ��^
�_ 
wres � � � � H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev � � � � H B E 0 D 3 4 8 D - 0 4 6 1 - 4 4 2 C - 8 2 5 E - 0 D 2 4 B 4 F D F 4 7 4
�` kfrmID    � �  ��]! �\ � b�[!�Z
�[ 
wres! �!! H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev!  �!! H 6 B C A 7 B E 4 - D 7 3 C - 4 3 9 4 - 8 9 8 8 - A D D B 0 B 1 B 1 2 C 4
�\ kfrmID   !! !�Y!�X! b�W!�V
�W 
wres! �!! H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev! �!	!	 H D 4 A B 1 A 5 C - D 4 6 5 - 4 A 5 1 - B 5 0 2 - 1 8 1 5 2 0 0 D 9 F 2 C
�X kfrmID   !
!
 !�U!�T! b�S!�R
�S 
wres! �!! H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev! �!! H B 8 5 E 7 2 3 2 - 9 5 F D - 4 9 F 2 - 9 D F A - B 9 C 2 7 4 6 6 B B E 1
�T kfrmID   !! !�Q!�P! b�O!�N
�O 
wres! �!! H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev! �!! H 5 C C A E 4 E B - F B A 7 - 4 B 7 6 - 8 A D 5 - 0 F 9 8 E 4 B 1 2 D 5 3
�P kfrmID   !! !�M!�L! b�K!�J
�K 
wres! �!! H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev! �!! H 3 6 0 B 2 0 3 3 - 2 0 5 4 - 4 C 0 A - B D 9 D - A 3 A 6 B E 3 9 3 6 9 8
�L kfrmID   !! !�I!�H! b�G!�F
�G 
wres! �! !  H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev! �!!!! H 2 5 F 5 8 5 4 9 - 6 1 8 0 - 4 4 2 7 - A 0 7 D - 7 1 6 A 9 6 F E 0 C A A
�H kfrmID   !"!" !#�E!$�D!# b�C!%�B
�C 
wres!% �!&!& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev!$ �!'!' H 6 F D F 6 F 2 9 - 1 7 E 1 - 4 1 6 3 - A F D A - 1 E 4 D C 2 2 5 C 4 9 0
�D kfrmID   !(!( !)�A!*�@!) b�?!+�>
�? 
wres!+ �!,!, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev!* �!-!- H A 6 4 5 D 0 3 9 - E 0 2 6 - 4 E 1 C - A A C E - 7 0 9 2 D F 6 E 3 C D B
�@ kfrmID   !.!. !/�=!0�<!/ b�;!1�:
�; 
wres!1 �!2!2 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev!0 �!3!3 H 2 5 F 5 8 5 4 9 - 6 1 8 0 - 4 4 2 7 - A 0 7 D - 7 1 6 A 9 6 F E 0 C A A
�< kfrmID   !4!4 !5�9!6�8!5 b�7!7�6
�7 
wres!7 �!8!8 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev!6 �!9!9 H 9 9 5 5 1 A 5 5 - C 1 C 4 - 4 0 D E - A D 0 6 - 1 B B E 9 6 3 7 7 1 1 7
�8 kfrmID   !:!: !;�5!<�4!; b�3!=�2
�3 
wres!= �!>!> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev!< �!?!? H 4 8 4 2 E 9 3 1 - B F 9 A - 4 5 6 8 - B 3 D D - 0 9 E 9 1 1 3 4 E 7 5 8
�4 kfrmID   !@!@ !A�1!B�0!A b�/!C�.
�/ 
wres!C �!D!D H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev!B �!E!E H C 9 A 9 2 3 E 1 - 5 B B 9 - 4 C B 2 - B 1 C 7 - 2 9 D 5 4 8 6 6 E D 6 5
�0 kfrmID   !F!F !G�-!H�,!G b�+!I�*
�+ 
wres!I �!J!J H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev!H �!K!K H 9 4 5 3 8 6 1 A - 7 E D E - 4 1 C 7 - A F 9 1 - E D 7 0 B 2 B C 4 3 8 0
�, kfrmID   !L!L !M�)!N�(!M b�'!O�&
�' 
wres!O �!P!P H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev!N �!Q!Q H 5 0 1 6 0 3 B 2 - 4 D D E - 4 E 2 1 - 9 6 7 5 - F 5 C C 2 2 F A 5 F 0 F
�( kfrmID   !R!R !S�%!T�$!S b�#!U�"
�# 
wres!U �!V!V H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev!T �!W!W H D 4 0 4 B 9 D D - 8 F 4 A - 4 3 6 B - A 3 5 6 - 2 A F 2 C 5 F 8 B E 5 C
�$ kfrmID   !X!X !Y�!!Z� !Y b�![�
� 
wres![ �!\!\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev!Z �!]!] H A D 4 C C A 0 B - 4 B 0 A - 4 5 B 6 - 8 A D 2 - D F 0 F 0 B 9 4 B 6 4 B
�  kfrmID   !^!^ !_�!`�!_ b�!a�
� 
wres!a �!b!b H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!` �!c!c H F D 2 9 3 4 9 7 - 9 1 8 9 - 4 2 4 2 - B 3 E 1 - D C C 4 D E 6 6 4 9 D 9
� kfrmID   !d!d !e�!f�!e b�!g�
� 
wres!g �!h!h H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!f �!i!i H 9 3 E 5 D 2 5 D - D D 1 2 - 4 0 C 8 - A 1 4 A - B 7 B A 8 5 5 1 7 6 C E
� kfrmID   !j!j !k�!l�!k b�!m�
� 
wres!m �!n!n H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!l �!o!o H 7 9 C E E 1 8 B - 5 A 0 E - 4 A 2 C - B 0 4 B - 0 B C D 5 5 3 3 F 7 C D
� kfrmID   !p!p !q�!r�!q b�!s�
� 
wres!s �!t!t H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!r �!u!u H F 6 D D A D 2 A - 5 1 2 8 - 4 2 B 5 - B 0 1 0 - 0 7 1 3 8 9 E 7 8 8 5 A
� kfrmID    !v!v !w�!x�!w b�!y�

� 
wres!y �!z!z H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev!x �!{!{ H 1 3 2 1 D 1 8 F - E 6 C A - 4 3 B 7 - B 0 F 1 - E 7 0 1 5 4 B C B E D B
� kfrmID  ! !|!| !}�	!~�!} b�!�
� 
wres! �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev!~ �!�!� H 2 8 8 3 D 8 B 6 - F 4 5 B - 4 E 6 F - 9 F 6 A - 9 9 4 1 B 9 B 4 F D 4 A
� kfrmID  " !�!� !��!��!� b�!��
� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!� �!�!� H 7 C D 8 8 9 B 7 - 7 4 A E - 4 D D B - B E 5 0 - 7 0 E 6 B 0 D B C 2 7 2
� kfrmID  # !�!� !��!�� !� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev!� �!�!� H 0 A E A 7 B 0 6 - 9 F 6 3 - 4 5 E B - 9 F B C - D 9 8 6 A 8 2 0 6 E C 9
�  kfrmID  $ !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 5 F 8 8 6 F 4 A - 1 6 F 2 - 4 1 5 A - 8 C B A - 3 2 9 B 8 D 3 E 4 0 5 9
�� kfrmID  % !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 5 B C 9 5 B A 6 - 4 E 7 D - 4 F 9 0 - A 2 5 C - 5 0 1 8 2 7 9 C 9 0 4 C
�� kfrmID  & !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H F C 1 4 B 9 E B - A 5 8 2 - 4 6 5 C - 8 A 6 E - D 8 1 3 D 4 A C 2 9 C 0
�� kfrmID  ' !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H C 1 B E 4 2 B 6 - 5 5 1 7 - 4 B 0 E - B F 1 7 - 7 C D 9 0 9 D F F 0 4 F
�� kfrmID  ( !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 6 D 1 4 5 6 C 7 - 6 F 0 8 - 4 E 3 2 - 9 5 8 A - 7 7 7 E 3 5 4 B D A A 3
�� kfrmID  ) !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 6 8 C 7 B B D 5 - 5 3 A 7 - 4 B E C - 9 B 5 8 - 1 F 5 B 2 3 5 0 1 F 9 A
�� kfrmID  * !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H A 9 F E 2 D D 6 - 8 1 0 9 - 4 F D 0 - 8 6 8 5 - C 6 0 1 F F 5 3 D 0 6 A
�� kfrmID  + !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 6 A 5 A 1 4 E 9 - 5 B 5 0 - 4 E 8 D - A 3 F F - 2 8 B E 9 2 E 5 E E A 5
�� kfrmID  , !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 2 D 7 2 5 3 6 1 - F 7 2 E - 4 9 E 7 - B 0 F C - 2 6 0 C E D 3 4 B E 9 D
�� kfrmID  - !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 9 4 5 6 9 3 7 3 - A E 2 5 - 4 4 8 6 - A 9 8 E - 0 4 8 6 F 4 F 1 4 6 B C
�� kfrmID  . !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H D 9 5 F 5 F B 3 - 2 F F D - 4 D E 0 - 8 8 D D - E 2 E 8 B 7 8 A A 6 5 3
�� kfrmID  / !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H B 0 2 5 D 3 0 5 - 2 B 3 C - 4 D 2 3 - 8 3 A A - 9 F 3 5 6 2 E B 8 4 D 1
�� kfrmID  0 !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 3 D A B 7 2 4 E - 6 8 3 8 - 4 D 2 E - A A E A - 5 E 1 7 3 E F 2 E A 9 3
�� kfrmID  1 !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 9 D D 7 A 5 0 7 - 4 7 E 5 - 4 8 F 4 - B E F 2 - 4 7 7 6 8 0 E E 0 7 1 4
�� kfrmID  2 !�!� !���!���!� b��!���
�� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev!� �!�!� H 8 E B 7 C 7 7 8 - 7 2 0 B - 4 6 8 C - 8 E 8 E - B 7 5 D 2 D C 9 9 F 0 D
�� kfrmID  3 !�!� !���!���!� b�!��
� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev!� �!�!� H B E 0 F 6 3 7 4 - C 2 1 1 - 4 B F 2 - A 6 3 5 - E 5 1 E E D 1 9 A F 6 A
�� kfrmID  4 !�!� !��!��!� b�!��
� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!� �!�!� H 2 A 2 F 7 5 1 7 - 0 9 0 4 - 4 1 9 3 - A B 2 D - 5 A 5 E D D 1 C A B 4 0
� kfrmID  5 !�!� !��!��!� b�!��
� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!� �!�!� H 3 C 8 F 6 1 3 C - 2 F 4 E - 4 9 3 9 - B F E E - D A 0 E C 8 1 E B 5 F 0
� kfrmID  6 !�!� !��!��!� b�!��
� 
wres!� �!�!� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev!� �!�!� H 6 E 3 E 6 3 C 3 - 6 4 2 9 - 4 F 5 D - A 3 F 2 - C D 2 4 1 0 2 2 F 2 6 8
� kfrmID  7 " "  "�"�" b�"�
� 
wres" �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev" �"" H 6 C 0 3 A 3 F 1 - 8 4 C 5 - 4 B E 8 - 8 2 D 9 - 2 A 0 2 2 9 8 1 9 D 5 4
� kfrmID  8 "" "�"�" b�"	�
� 
wres"	 �"
"
 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev" �"" H E E 2 F 7 C 1 6 - 7 5 2 1 - 4 2 B 3 - A 1 A 0 - 2 6 0 4 1 9 0 B B 3 E F
� kfrmID  9 "" "�"�" b�"�
� 
wres" �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev" �"" H 5 D F C 7 6 E D - D F 1 D - 4 3 7 1 - 8 4 1 D - 1 6 2 4 D 2 7 5 9 4 E 1
� kfrmID  : "" "�"�" b�"�
� 
wres" �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev" �"" H 3 C 4 7 7 5 2 5 - 4 7 7 C - 4 B B 5 - 9 7 E 5 - 6 D 0 8 6 2 9 C 8 2 E D
� kfrmID  ; "" "�"�" b�"�
� 
wres" �"" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev" �"" H B 2 B 4 8 9 0 0 - D 8 A B - 4 D 7 A - 8 F 7 5 - 5 2 C A 5 C 0 B 1 1 9 4
� kfrmID  < "" "�" �" b�"!�
� 
wres"! �"""" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"  �"#"# H A 4 2 6 5 1 C B - D 4 3 F - 4 C C A - 9 3 8 E - 9 8 8 3 A E 0 E 1 3 A E
� kfrmID  = "$"$ "%�"&�"% b�"'�
� 
wres"' �"("( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"& �")") H E C E 6 F 8 C 5 - E 0 1 F - 4 A 2 C - 8 3 0 D - 8 9 7 3 E 4 A 4 4 0 5 6
� kfrmID  > "*"* "+�",�"+ b�"-�
� 
wres"- �".". H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev", �"/"/ H 3 9 A 6 0 9 E 6 - A A E B - 4 9 7 D - 8 E 1 E - C 2 6 9 C E F 0 D B 0 0
� kfrmID  ? "0"0 "1�"2�"1 b�"3�
� 
wres"3 �"4"4 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"2 �"5"5 H E 1 5 F 5 D 7 8 - D 9 B 7 - 4 9 A 4 - 9 3 F 6 - 6 7 C 2 9 B 0 9 8 3 1 A
� kfrmID  @ "6"6 "7�"8�"7 b�"9�
� 
wres"9 �":": H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"8 �";"; H E 6 D F 3 A 6 2 - 2 7 E 0 - 4 2 B 6 - B 0 B D - D 8 A 6 E 9 4 4 A E 7 A
� kfrmID  A "<"< "=�">�"= b�"?�
� 
wres"? �"@"@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"> �"A"A H 7 A 9 F D 7 5 E - D F 9 B - 4 9 1 7 - B A 4 3 - E 6 B 9 7 D 3 F E 6 E 6
� kfrmID  B "B"B "C�"D�"C b�"E�
� 
wres"E �"F"F H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"D �"G"G H A 6 8 4 5 2 6 5 - 9 B A 7 - 4 2 4 F - B A 4 B - 3 5 F 0 1 2 F 5 4 2 6 8
� kfrmID  C "H"H "I�"J�"I b�"K�~
� 
wres"K �"L"L H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev"J �"M"M H C 2 1 2 3 B 8 7 - 9 C 9 1 - 4 3 1 A - B 4 D A - A 1 C 1 2 E F A A 9 6 3
� kfrmID  D "N"N "O�}"P�|"O b�{"Q�z
�{ 
wres"Q �"R"R H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev"P �"S"S H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�| kfrmID  E "T"T "U�y"V�x"U b�w"W�v
�w 
wres"W �"X"X H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev"V �"Y"Y H C 4 C 4 D 5 B 8 - F 2 5 8 - 4 5 C A - B 5 8 5 - E 5 1 4 C E B C B 5 2 1
�x kfrmID  F "Z"Z "[�u"\�t"[ b�s"]�r
�s 
wres"] �"^"^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev"\ �"_"_ H 3 9 A 0 A F 2 8 - E C 2 4 - 4 8 1 A - A 6 B 7 - 7 B 5 9 4 B 7 A 0 8 0 E
�t kfrmID  G "`"` "a�q"b�p"a b�o"c�n
�o 
wres"c �"d"d H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev"b �"e"e H 2 5 9 9 7 E 1 7 - 1 E 4 F - 4 5 D 9 - 8 D F B - C B A F 6 1 3 C 6 E C 6
�p kfrmID  H "f"f "g�m"h�l"g b�k"i�j
�k 
wres"i �"j"j H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev"h �"k"k H 9 F B 2 E 0 3 9 - 3 A F D - 4 6 4 0 - B 7 4 6 - 5 9 3 9 9 4 9 B D 1 3 5
�l kfrmID  I "l"l "m�i"n�h"m b�g"o�f
�g 
wres"o �"p"p H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev"n �"q"q H 6 8 A 0 8 4 9 F - E 5 A F - 4 5 2 A - 9 C 3 4 - 6 7 3 7 2 F D B 2 E E C
�h kfrmID  J "r"r "s�e"t�d"s b�c"u�b
�c 
wres"u �"v"v H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev"t �"w"w H 1 3 8 7 C E 3 5 - 5 7 5 D - 4 3 0 9 - 9 F 0 B - 4 F 6 7 E 5 9 A 0 1 7 5
�d kfrmID  K "x"x "y�a"z�`"y b�_"{�^
�_ 
wres"{ �"|"| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev"z �"}"} H E 4 C A 1 B 0 5 - 4 9 B 8 - 4 7 4 3 - B 5 6 2 - 2 A 9 5 0 A 1 2 3 9 6 A
�` kfrmID  L "~"~ "�]"��\" b�["��Z
�[ 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev"� �"�"� H E 4 E A 8 A 5 D - 2 F 9 5 - 4 9 4 8 - 8 C 3 A - 9 1 D 1 E E 6 B 6 1 7 2
�\ kfrmID  M "�"� "��Y"��X"� b�W"��V
�W 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev"� �"�"� H 4 A F 5 F 3 E 5 - 0 2 1 9 - 4 9 2 2 - A E 7 E - 1 B 9 B 6 E D D 2 0 A 0
�X kfrmID  N "�"� "��U"��T"� b�S"��R
�S 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev"� �"�"� H 6 A 7 D E A 1 5 - 7 3 7 4 - 4 D F A - A E B B - 2 7 9 A 8 6 5 D A 7 D E
�T kfrmID  O "�"� "��Q"��P"� b�O"��N
�O 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev"� �"�"� H 2 5 9 9 7 E 1 7 - 1 E 4 F - 4 5 D 9 - 8 D F B - C B A F 6 1 3 C 6 E C 6
�P kfrmID  P "�"� "��M"��L"� b�K"��J
�K 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev"� �"�"� H 7 7 F 6 7 9 6 F - D 4 4 0 - 4 3 3 F - A 2 C D - F 0 B 6 9 7 D 2 5 C 2 A
�L kfrmID  Q "�"� "��I"��H"� b�G"��F
�G 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev"� �"�"� H 8 7 5 0 2 5 C B - 0 5 5 7 - 4 D 4 E - B B 4 5 - 0 9 D 0 F 1 6 8 B 2 5 B
�H kfrmID  R "�"� "��E"��D"� b�C"��B
�C 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev"� �"�"� H A E B 0 B 5 F F - 5 9 B 9 - 4 8 7 6 - 8 B C E - 0 9 2 D F 8 D 6 3 2 8 6
�D kfrmID  S "�"� "��A"��@"� b�?"��>
�? 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev"� �"�"� H 4 C 7 0 C 1 B 5 - 8 E 7 6 - 4 F B 1 - 9 A 9 8 - 6 A 6 4 6 C D F C 1 9 E
�@ kfrmID  T "�"� "��="��<"� b�;"��:
�; 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev"� �"�"� H A D 1 8 C 4 A A - 7 A C 8 - 4 E E 8 - 8 5 6 3 - 3 F 3 0 8 A 6 1 0 B 9 3
�< kfrmID  U "�"� "��9"��8"� b�7"��6
�7 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev"� �"�"� H 4 F E 5 4 9 0 E - 0 7 1 8 - 4 8 7 1 - 9 F 6 3 - B A 7 1 F 2 3 8 8 5 5 C
�8 kfrmID  V "�"� "��5"��4"� b�3"��2
�3 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev"� �"�"� H 6 A 7 D E A 1 5 - 7 3 7 4 - 4 D F A - A E B B - 2 7 9 A 8 6 5 D A 7 D E
�4 kfrmID  W "�"� "��1"��0"� b�/"��.
�/ 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev"� �"�"� H 7 F 6 6 8 7 3 D - 0 4 9 9 - 4 3 1 B - 8 6 3 C - 7 7 F E 7 A 1 0 9 6 2 0
�0 kfrmID  X "�"� "��-"��,"� b�+"��*
�+ 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev"� �"�"� H 7 D 3 A E 3 8 3 - F 3 2 6 - 4 1 C 9 - A 4 E 1 - 1 D E 6 8 C 4 4 4 1 3 C
�, kfrmID  Y "�"� "��)"��("� b�'"��&
�' 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev"� �"�"� H B 5 B 0 4 7 B 3 - D C 7 3 - 4 F F 0 - 9 7 D 3 - E F D 5 C 7 4 3 2 A 5 0
�( kfrmID  Z "�"� "��%"��$"� b�#"��"
�# 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev"� �"�"� H 2 C C A 3 F B 4 - 1 E B D - 4 D B 2 - B 7 8 9 - 8 8 1 3 6 7 A A D B B 5
�$ kfrmID  [ "�"� "��!"�� "� b�"��
� 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev"� �"�"� H B 6 4 C 1 5 E 0 - 5 7 0 6 - 4 1 D 5 - B 0 E E - 7 5 D F 7 F 5 1 2 D 7 2
�  kfrmID  \ "�"� "��"��"� b�"��
� 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"� �"�"� H 5 B 0 E C E 5 A - 3 7 1 7 - 4 3 1 9 - 8 C 8 C - 9 A 5 5 D F A 6 A A 3 3
� kfrmID  ] "�"� "��"��"� b�"��
� 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"� �"�"� H C 8 8 D 4 5 B F - 6 8 A 0 - 4 B 9 0 - 9 C 4 3 - 6 E 3 0 A 2 E 0 C 3 6 E
� kfrmID  ^ "�"� "��"��"� b�"��
� 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"� �"�"� H F 0 5 3 F E 9 6 - F 1 3 7 - 4 3 3 A - A 8 0 F - 9 A B 7 A A 4 F 8 0 1 7
� kfrmID  _ "�"� "��"��"� b�"��
� 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev"� �"�"� H 8 3 C 9 7 5 D 2 - B E 0 3 - 4 1 8 0 - 9 A A 2 - 3 4 0 1 B 9 F A 7 F 8 A
� kfrmID  ` "�"� "��"��"� b�"��

� 
wres"� �"�"� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev"� �"�"� H C F C 9 2 C E E - F A B 2 - 4 6 2 4 - 9 8 B E - B 0 F 7 0 0 B 2 4 9 F 8
� kfrmID  a "�"� "��	"��"� b�"��
� 
wres"� �# #  H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev"� �## H 3 B C 7 A A 3 8 - 8 E 9 9 - 4 D 5 8 - B 9 6 7 - D 1 6 3 0 4 D B 0 7 2 D
� kfrmID  b ## #�#�# b�#�
� 
wres# �## H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev# �## H B C 5 9 2 5 E C - 1 8 0 8 - 4 6 8 B - B B C C - 8 5 C 2 4 5 D 9 D 0 5 6
� kfrmID  c ## #	�#
� #	 b��#��
�� 
wres# �## H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev#
 �## H 8 A B 9 6 F D B - D C C 1 - 4 9 E 9 - 8 F 3 4 - 4 8 A A 2 8 5 2 1 8 8 B
�  kfrmID  d ## #��#��# b��#��
�� 
wres# �## H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev# �## H 3 6 3 0 4 2 0 D - 8 0 3 6 - 4 C 9 A - 9 A 1 B - E 2 E 4 D D 9 0 E 1 F 5
�� kfrmID  e ## #��#��# b��#��
�� 
wres# �## H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev# �## H 4 9 C F 8 0 0 5 - E D 0 3 - 4 A E A - 8 3 1 3 - 4 0 C 1 1 9 3 6 2 0 C F
�� kfrmID  f ## #��#��# b��#��
�� 
wres# �## H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev# �## H C 2 E C 7 B D A - D B 6 2 - 4 D 9 9 - A 5 4 0 - B B 6 B 5 1 6 D 0 9 A 4
�� kfrmID  g # #  #!��#"��#! b��##��
�� 
wres## �#$#$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#" �#%#% H F 5 B B 5 8 4 4 - 8 5 F 5 - 4 D F 4 - 9 0 B D - 8 A B 4 1 F 6 4 7 F C D
�� kfrmID  h #&#& #'��#(��#' b��#)��
�� 
wres#) �#*#* H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#( �#+#+ H 3 A 1 9 E 0 F 3 - 2 3 8 2 - 4 8 0 5 - 8 4 E 6 - 3 A 0 3 0 B 2 0 3 C C 1
�� kfrmID  i #,#, #-��#.��#- b��#/��
�� 
wres#/ �#0#0 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#. �#1#1 H B 4 7 C 5 0 3 8 - 4 1 E 8 - 4 D A F - 8 F 6 E - C A 1 3 6 F 0 F 2 6 0 D
�� kfrmID  j #2#2 #3��#4��#3 b��#5��
�� 
wres#5 �#6#6 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#4 �#7#7 H 1 E 9 8 F 9 F E - 9 9 8 6 - 4 D B 0 - 8 5 0 2 - 7 F B 8 0 8 4 0 6 D 7 F
�� kfrmID  k #8#8 #9��#:��#9 b��#;��
�� 
wres#; �#<#< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#: �#=#= H A 3 8 A B D C 5 - 4 3 B 8 - 4 8 F 7 - 9 8 1 6 - 1 0 0 2 C 9 6 0 7 1 6 F
�� kfrmID  l #>#> #?��#@��#? b��#A��
�� 
wres#A �#B#B H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#@ �#C#C H 5 E 1 7 D 5 E 3 - 4 D E F - 4 8 0 6 - B 4 A C - D F 2 0 C 9 A F 4 0 3 9
�� kfrmID  m #D#D #E��#F��#E b��#G��
�� 
wres#G �#H#H H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#F �#I#I H A F 2 9 B 1 D 9 - 5 7 D 2 - 4 2 4 3 - B 1 5 4 - E 5 3 7 5 9 1 1 B 8 B 5
�� kfrmID  n #J#J #K��#L��#K b��#M��
�� 
wres#M �#N#N H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#L �#O#O H 9 6 C B 2 4 F E - 3 8 2 4 - 4 B B E - 8 A A 1 - B 7 B 4 7 E 0 C A 4 1 F
�� kfrmID  o #P#P #Q��#R��#Q b��#S��
�� 
wres#S �#T#T H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#R �#U#U H 8 4 F C C 6 1 7 - 7 1 7 A - 4 2 4 A - 8 0 3 3 - 6 D 1 F 3 7 2 9 5 8 0 1
�� kfrmID  p #V#V #W��#X��#W b��#Y��
�� 
wres#Y �#Z#Z H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#X �#[#[ H 9 8 1 F 1 5 4 0 - 8 2 5 D - 4 0 1 C - B 3 C B - C 5 7 C 1 0 9 A 3 9 4 1
�� kfrmID  q #\#\ #]��#^��#] b��#_��
�� 
wres#_ �#`#` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#^ �#a#a H B D 8 3 0 9 9 1 - 2 1 2 D - 4 E D F - 8 0 D 3 - 4 7 2 2 A B F 0 5 0 A 6
�� kfrmID  r #b#b #c��#d��#c b��#e��
�� 
wres#e �#f#f H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev#d �#g#g H 6 3 A 6 7 0 E F - 9 8 8 2 - 4 0 F F - 9 0 E F - 8 1 C E F 3 4 9 9 7 C 1
�� kfrmID  s #h#h #i��#j��#i b�#k�
� 
wres#k �#l#l H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev#j �#m#m H C 4 8 4 1 7 3 D - 7 8 F C - 4 A D 9 - 9 C 1 7 - 7 4 1 D 2 3 7 F E C 5 4
�� kfrmID  t #n#n #o�#p�#o b�#q�
� 
wres#q �#r#r H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#p �#s#s H F A 0 3 2 8 7 C - 9 F 9 8 - 4 F D 5 - A A 6 A - 9 2 3 D 9 9 4 F 1 6 D 0
� kfrmID  u #t#t #u�#v�#u b�#w�
� 
wres#w �#x#x H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#v �#y#y H 3 9 8 4 B C 4 A - 9 2 9 B - 4 6 9 5 - A 4 3 9 - 2 3 9 8 E A C C F 9 0 6
� kfrmID  v #z#z #{�#|�#{ b�#}�
� 
wres#} �#~#~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#| �## H C E 2 3 0 5 6 2 - 7 9 9 F - 4 1 B F - 8 8 B 7 - F E 0 1 7 E D 9 8 D 1 0
� kfrmID  w #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 3 A 0 D B 8 A 8 - 3 F 9 3 - 4 0 2 5 - 8 E 7 2 - 8 9 C 2 8 4 E 9 0 8 C 4
� kfrmID  x #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 2 0 5 0 2 0 5 7 - 4 2 E F - 4 3 2 5 - 9 9 B E - 1 7 5 C 4 2 A 7 6 1 7 7
� kfrmID  y #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 9 7 C 2 F 8 B E - 9 7 D 1 - 4 2 4 F - 9 2 3 5 - 9 9 5 A 2 B 7 0 7 0 8 2
� kfrmID  z #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 2 2 9 8 F 9 9 4 - 3 3 8 7 - 4 0 4 D - 9 4 5 7 - C 8 A A B 8 5 B 2 8 D 6
� kfrmID  { #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H D 2 C 8 7 2 E 2 - 8 9 5 F - 4 7 7 6 - B 6 2 7 - 8 6 9 D B 1 3 E 2 1 8 7
� kfrmID  | #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H D 9 B 7 F E 5 6 - C 8 5 A - 4 D E 3 - 8 7 E E - C 7 5 4 B B 2 4 4 9 C 6
� kfrmID  } #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H F 7 F 4 2 2 F 4 - B B 0 E - 4 B 7 C - 9 1 8 9 - 6 7 3 B E 8 6 5 6 5 2 3
� kfrmID  ~ #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H D A 6 F 7 4 E 8 - 0 A B 5 - 4 4 C 8 - 8 8 3 4 - 1 D 2 0 A D D A 0 6 9 5
� kfrmID   #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 4 6 5 F 6 4 A 3 - E 5 D A - 4 8 2 D - B 4 8 B - D 3 6 9 A 3 9 1 5 7 4 4
� kfrmID  � #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 6 B 5 0 3 6 5 7 - 3 C B A - 4 9 C 5 - A E 9 8 - 4 1 C A B 6 9 B 1 F 3 8
� kfrmID  � #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H E 6 C 8 1 C 5 F - 6 7 6 D - 4 F 1 7 - A B 5 E - A C 8 E 7 3 E 2 3 9 1 D
� kfrmID  � #�#� #��#��#� b�#��
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev#� �#�#� H 7 4 9 1 3 9 9 5 - 4 B F F - 4 2 0 1 - 9 7 7 9 - F 1 8 3 3 0 E D 7 0 1 1
� kfrmID  � #�#� #��#��#� b�#��~
� 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev#� �#�#� H F B 7 A 3 9 A 9 - F 6 C C - 4 F 2 6 - 9 4 8 2 - E 7 3 2 5 C 3 7 5 F 8 5
� kfrmID  � #�#� #��}#��|#� b�{#��z
�{ 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev#� �#�#� H 2 4 D B B 1 E 9 - D F D 6 - 4 C 8 6 - B 8 0 7 - 4 A 5 0 3 C C C 2 B 0 6
�| kfrmID  � #�#� #��y#��x#� b�w#��v
�w 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev#� �#�#� H 5 A 4 9 1 C 3 2 - 1 B 3 B - 4 D C 7 - 9 0 7 5 - 4 5 D 9 2 1 1 7 5 2 2 9
�x kfrmID  � #�#� #��u#��t#� b�s#��r
�s 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev#� �#�#� H 4 F 2 F 6 9 A 4 - C F 0 F - 4 9 6 4 - 8 D 5 3 - 6 A 1 C 2 3 5 A 9 6 6 E
�t kfrmID  � #�#� #��q#��p#� b�o#��n
�o 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev#� �#�#� H B B B 5 E B D 2 - 9 9 5 9 - 4 7 0 E - A A 7 B - 6 E 2 0 4 3 7 7 6 E 5 3
�p kfrmID  � #�#� #��m#��l#� b�k#��j
�k 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev#� �#�#� H 8 0 A 5 B A 9 2 - B 0 E F - 4 D E D - A 5 4 8 - 7 0 4 2 4 9 D D 4 F E 5
�l kfrmID  � #�#� #��i#��h#� b�g#��f
�g 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev#� �#�#� H 0 B 9 8 1 C B 4 - D A B B - 4 1 F 3 - A E D 2 - 2 B 2 1 7 4 8 E 3 8 5 6
�h kfrmID  � #�#� #��e#��d#� b�c#��b
�c 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev#� �#�#� H 1 7 D 5 B B 5 D - 0 B 7 1 - 4 7 F 5 - 8 8 A 5 - C 3 D 6 A A 6 9 3 0 8 5
�d kfrmID  � #�#� #��a#��`#� b�_#��^
�_ 
wres#� �#�#� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev#� �#�#� H B 3 E 4 5 F E C - 6 3 D 5 - 4 1 2 4 - B 4 3 3 - B 7 0 0 3 9 A 9 6 D 1 3
�` kfrmID  � #�#� #��]$ �\#� b�[$�Z
�[ 
wres$ �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev$  �$$ H 6 A 1 3 9 C F 2 - 0 D 7 B - 4 5 8 2 - 9 4 2 D - 9 E 2 F 9 B 4 4 F 1 4 6
�\ kfrmID  � $$ $�Y$�X$ b�W$�V
�W 
wres$ �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev$ �$	$	 H 1 1 5 B C 4 0 1 - 2 9 6 3 - 4 5 4 3 - 8 A 9 1 - 7 5 2 B A A 3 4 0 B 9 8
�X kfrmID  � $
$
 $�U$�T$ b�S$�R
�S 
wres$ �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev$ �$$ H D 0 D C D B 1 F - 3 4 A 1 - 4 9 E 5 - A 8 E 7 - F E F C 6 C 4 6 2 A 8 9
�T kfrmID  � $$ $�Q$�P$ b�O$�N
�O 
wres$ �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev$ �$$ H 4 1 8 D 8 0 3 5 - 9 9 2 7 - 4 7 9 5 - 9 B B F - D 7 0 D 7 7 0 1 9 A E A
�P kfrmID  � $$ $�M$�L$ b�K$�J
�K 
wres$ �$$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev$ �$$ H E D F 1 9 C 5 5 - 7 F 3 7 - 4 7 3 5 - A 7 F 9 - 2 4 1 3 E F C 2 6 8 A 0
�L kfrmID  � $$ $�I$�H$ b�G$�F
�G 
wres$ �$ $  H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev$ �$!$! H F C 0 D D 5 3 9 - 0 2 2 6 - 4 3 B 7 - 8 3 A 1 - 6 5 F 7 1 7 C 0 0 1 C 5
�H kfrmID  � $"$" $#�E$$�D$# b�C$%�B
�C 
wres$% �$&$& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev$$ �$'$' H 5 8 F 9 6 0 5 0 - 0 B 3 4 - 4 B A D - 8 0 6 B - B E 2 F 6 D 6 F B B 5 9
�D kfrmID  � $($( $)�A$*�@$) b�?$+�>
�? 
wres$+ �$,$, H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev$* �$-$- H 2 5 F 5 8 5 4 9 - 6 1 8 0 - 4 4 2 7 - A 0 7 D - 7 1 6 A 9 6 F E 0 C A A
�@ kfrmID  � $.$. $/�=$0�<$/ b�;$1�:
�; 
wres$1 �$2$2 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev$0 �$3$3 H 6 3 E B B 2 5 E - E 5 9 5 - 4 7 E D - 8 0 A 5 - 5 9 9 9 8 E 6 3 1 4 5 2
�< kfrmID  � $4$4 $5�9$6�8$5 b�7$7�6
�7 
wres$7 �$8$8 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev$6 �$9$9 H 0 1 7 3 5 4 1 3 - 9 7 5 1 - 4 3 4 0 - B A 8 B - E 5 F D 8 D 3 5 3 0 8 5
�8 kfrmID  � $:$: $;�5$<�4$; b�3$=�2
�3 
wres$= �$>$> H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev$< �$?$? H 7 0 2 7 3 C A D - 5 F 9 D - 4 A 4 A - A 8 C F - 9 2 7 2 F A E D C 5 E 6
�4 kfrmID  � $@$@ $A�1$B�0$A b�/$C�.
�/ 
wres$C �$D$D H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev$B �$E$E H 7 1 1 4 0 3 1 6 - 7 6 E A - 4 2 9 5 - A F 2 3 - 6 6 9 5 9 4 6 D 9 8 4 9
�0 kfrmID  � $F$F $G�-$H�,$G b�+$I�*
�+ 
wres$I �$J$J H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev$H �$K$K H 5 5 8 D 5 9 0 8 - A 7 9 C - 4 1 8 9 - A E 1 9 - B D 3 3 A 4 E D 9 2 F C
�, kfrmID  � $L$L $M�)$N�($M b�'$O�&
�' 
wres$O �$P$P H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev$N �$Q$Q H C 5 5 3 7 8 6 3 - 6 3 1 7 - 4 B A 0 - 8 F B 8 - 4 5 A A A 1 9 7 2 3 8 E
�( kfrmID  � $R$R $S�%$T�$$S b�#$U�"
�# 
wres$U �$V$V H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev$T �$W$W H 0 5 2 D 7 D 1 B - F F 3 0 - 4 E 4 3 - A F D E - D 6 1 F 3 3 9 8 E 5 F 7
�$ kfrmID  � $X$X $Y�!$Z� $Y b�$[�
� 
wres$[ �$\$\ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev$Z �$]$] H 9 C 2 3 0 E 3 D - 6 4 9 7 - 4 2 4 0 - 8 8 0 C - 8 1 4 F F 1 0 D C F 5 8
�  kfrmID  � $^$^ $_�$`�$_ b�$a�
� 
wres$a �$b$b H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$` �$c$c H 9 C 2 3 0 E 3 D - 6 4 9 7 - 4 2 4 0 - 8 8 0 C - 8 1 4 F F 1 0 D C F 5 8
� kfrmID  � $d$d $e�$f�$e b�$g�
� 
wres$g �$h$h H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$f �$i$i H 9 C 2 3 0 E 3 D - 6 4 9 7 - 4 2 4 0 - 8 8 0 C - 8 1 4 F F 1 0 D C F 5 8
� kfrmID  � $j$j $k�$l�$k b�$m�
� 
wres$m �$n$n H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$l �$o$o H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
� kfrmID  � $p$p $q�$r�$q b�$s�
� 
wres$s �$t$t H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$r �$u$u H B 3 A 6 7 F 7 B - 9 E 5 A - 4 3 7 4 - B F A 7 - C 2 C 5 6 0 7 F 5 9 0 9
� kfrmID  � $v$v $w�$x�$w b�$y�

� 
wres$y �$z$z H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev$x �${${ H 4 8 A 8 3 4 F B - 9 9 2 D - 4 3 D B - B 0 9 F - 0 5 4 1 D 7 0 0 0 2 E 4
� kfrmID  � $|$| $}�	$~�$} b�$�
� 
wres$ �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev$~ �$�$� H 3 4 1 6 5 3 9 E - 8 A A 7 - 4 A 4 6 - 9 6 8 7 - 7 A 2 9 9 0 F A C 9 B 8
� kfrmID  � $�$� $��$��$� b�$��
� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$� �$�$� H B 8 4 D 2 5 F 1 - 2 D 4 1 - 4 8 5 6 - B A B C - 7 5 5 F 6 B 2 9 6 E A E
� kfrmID  � $�$� $��$�� $� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev$� �$�$� H 8 E 1 A 3 1 1 B - B E 4 1 - 4 D C 7 - 8 2 6 5 - 9 3 A 8 9 F 4 D E F 7 0
�  kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 8 E 1 A 3 1 1 B - B E 4 1 - 4 D C 7 - 8 2 6 5 - 9 3 A 8 9 F 4 D E F 7 0
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 3 7 0 F D 6 A C - 1 C 1 3 - 4 2 C 0 - B 7 C 7 - 0 D 3 7 3 1 7 E A C 6 7
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 2 C C A 3 F B 4 - 1 E B D - 4 D B 2 - B 7 8 9 - 8 8 1 3 6 7 A A D B B 5
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 1 4 0 3 2 C 4 D - 7 1 F 9 - 4 D 5 C - B 0 4 6 - 2 C 8 E 5 1 F 0 A A 5 F
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H F 0 5 B 2 0 E 3 - 4 0 7 E - 4 A 4 2 - B 5 0 0 - 4 3 3 E 4 6 6 2 2 A 6 B
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 2 D 9 5 E 8 4 9 - C F B D - 4 5 4 4 - A 5 0 C - A 0 E 1 F A 0 4 4 D 2 0
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 2 5 F 5 8 5 4 9 - 6 1 8 0 - 4 4 2 7 - A 0 7 D - 7 1 6 A 9 6 F E 0 C A A
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 2 5 F 5 8 5 4 9 - 6 1 8 0 - 4 4 2 7 - A 0 7 D - 7 1 6 A 9 6 F E 0 C A A
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H A B 8 0 1 9 A 5 - 4 F 5 2 - 4 D F E - 8 E 9 4 - 8 6 E F B 8 1 A 0 3 A 6
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H D 0 5 7 A C 8 6 - 4 8 E B - 4 2 D 5 - B 4 3 3 - 3 9 1 0 1 A 7 8 A D B 6
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 3 7 2 6 D 5 3 3 - 9 0 4 D - 4 B 2 B - B 1 9 4 - 2 E 1 9 4 4 5 3 6 2 F B
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H E 0 C 7 2 A C 5 - F 0 E 3 - 4 C A 8 - A C 5 3 - F 0 8 B 1 1 C E B B A 7
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 6 C 9 4 B 0 E 3 - B 5 E 0 - 4 0 6 C - A D F F - B 5 7 7 A E 1 E 2 F 8 C
�� kfrmID  � $�$� $���$���$� b��$���
�� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev$� �$�$� H 1 3 E 2 B A 8 F - 9 F F 0 - 4 3 A B - 9 7 0 C - 0 D 2 4 D 8 C A F 5 E 7
�� kfrmID  � $�$� $���$���$� b�$��
� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev$� �$�$� H 7 5 3 3 E A E B - B 5 6 C - 4 9 5 B - B 8 0 7 - 9 6 1 E C 4 8 5 3 5 B 2
�� kfrmID  � $�$� $��$��$� b�$��
� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$� �$�$� H A 1 7 4 A 5 C C - 0 4 1 2 - 4 4 4 2 - 9 D 7 E - C 1 D A E 4 0 D A 1 E C
� kfrmID  � $�$� $��$��$� b�$��
� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$� �$�$� H 0 0 1 E 3 6 8 4 - 5 1 3 9 - 4 0 5 D - B 6 0 1 - 7 1 F B 7 7 5 4 2 A 2 B
� kfrmID  � $�$� $��$��$� b�$��
� 
wres$� �$�$� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev$� �$�$� H F 2 2 1 C C 7 7 - 9 B C 1 - 4 3 A D - 8 2 A B - 8 5 0 D E F 6 A B D F 2
� kfrmID  � % %  %�%�% b�%�
� 
wres% �%% H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev% �%% H 5 7 3 8 4 3 4 4 - 7 0 8 C - 4 B C A - A 7 B 9 - 4 9 6 2 E 3 F 1 6 B 0 5
� kfrmID  � %% %�%�% b�%	�
� 
wres%	 �%
%
 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev% �%% H 2 6 3 1 4 E C 8 - F 8 C A - 4 D F 7 - B 5 B 6 - 7 C 0 A 7 3 E 1 D 9 E B
� kfrmID  � %% %�%�% b�%�
� 
wres% �%% H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev% �%% H 4 9 6 0 2 4 E C - D 3 9 4 - 4 B 5 0 - A 4 1 C - B E B E 9 1 2 B 8 E 5 9
� kfrmID  � %% %�%�% b�%�
� 
wres% �%% H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev% �%% H 1 D 5 E A 0 4 2 - E 2 3 4 - 4 F 2 D - B B B F - 9 A 5 B 7 4 E 6 0 4 7 1
� kfrmID  � %% %�%�% b�%�
� 
wres% �%% H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev% �%% H 9 9 D 6 B 2 2 A - 7 2 C B - 4 C 3 A - 8 F 9 C - C 1 F 2 B E C 4 B 9 0 B
� kfrmID  � %% %�% �% b�%!�
� 
wres%! �%"%" H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%  �%#%# H 6 8 6 5 E 6 5 C - A A A A - 4 3 0 3 - 8 B C 6 - 6 F 6 B 7 8 3 2 D 9 1 8
� kfrmID  � %$%$ %%�%&�%% b�%'�
� 
wres%' �%(%( H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%& �%)%) H F 9 5 F D 0 C 5 - 7 E 0 2 - 4 C 0 5 - B D 6 8 - A 2 6 2 7 7 E 1 5 5 9 3
� kfrmID  � %*%* %+�%,�%+ b�%-�
� 
wres%- �%.%. H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%, �%/%/ H 3 7 8 B 0 C 7 1 - 7 1 6 0 - 4 6 7 A - B 1 E 1 - 6 E 0 F 0 4 B 6 D 3 D E
� kfrmID  � %0%0 %1�%2�%1 b�%3�
� 
wres%3 �%4%4 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%2 �%5%5 H 1 A 1 9 4 8 F 7 - 0 E 0 B - 4 8 D A - A 5 E E - 4 C 6 A A A 9 2 B 8 A 1
� kfrmID  � %6%6 %7�%8�%7 b�%9�
� 
wres%9 �%:%: H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%8 �%;%; H 0 5 2 C 0 4 3 1 - A 1 C 1 - 4 2 8 1 - A F 1 C - 1 3 C 0 6 9 A E 4 4 5 B
� kfrmID  � %<%< %=�%>�%= b�%?�
� 
wres%? �%@%@ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%> �%A%A H 1 8 B 9 A 1 4 8 - 1 A A B - 4 F C 4 - A D 3 E - B C C 4 2 B E 3 9 F 0 3
� kfrmID  � %B%B %C�%D�%C b�%E�
� 
wres%E �%F%F H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%D �%G%G H 8 2 7 8 C 2 E 8 - 5 9 0 C - 4 2 C C - 8 5 F F - 4 6 6 9 E 4 4 6 8 A A C
� kfrmID  � %H%H %I�%J�%I b�%K�~
� 
wres%K �%L%L H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev%J �%M%M H 3 2 9 A C 1 B F - 9 A 0 0 - 4 7 F D - B C 1 3 - 0 C B 9 0 5 D E E 4 F D
� kfrmID  � %N%N %O�}%P�|%O b�{%Q�z
�{ 
wres%Q �%R%R H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev%P �%S%S H 1 C C 4 0 8 C 6 - E 9 D 1 - 4 D 2 4 - B 2 1 F - 7 1 D D 5 9 6 5 3 5 9 0
�| kfrmID  � %T%T %U�y%V�x%U b�w%W�v
�w 
wres%W �%X%X H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev%V �%Y%Y H F D 4 F B D 1 E - C D 3 6 - 4 2 9 C - B 4 2 9 - F 5 4 5 7 2 3 D 4 9 A A
�x kfrmID  � %Z%Z %[�u%\�t%[ b�s%]�r
�s 
wres%] �%^%^ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev%\ �%_%_ H C 1 3 A 4 B A 7 - E 4 C 1 - 4 B D F - 8 6 2 E - 4 2 C 7 0 4 D F 8 7 E 5
�t kfrmID  � %`%` %a�q%b�p%a b�o%c�n
�o 
wres%c �%d%d H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev%b �%e%e H 6 8 E B 7 F 5 8 - 7 5 8 9 - 4 D 6 5 - 9 1 3 E - 5 5 0 7 1 8 5 D 7 E 7 3
�p kfrmID  � %f%f %g�m%h�l%g b�k%i�j
�k 
wres%i �%j%j H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev%h �%k%k H 9 6 7 6 3 D 9 2 - 0 8 F 2 - 4 D 6 B - A 4 F 3 - 3 6 E B 3 5 6 D 8 0 2 1
�l kfrmID  � %l%l %m�i%n�h%m b�g%o�f
�g 
wres%o �%p%p H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev%n �%q%q H 9 D 7 6 7 4 D F - 7 5 8 D - 4 A 4 3 - 8 F B 6 - F C 7 9 E 7 6 2 7 0 3 5
�h kfrmID  � %r%r %s�e%t�d%s b�c%u�b
�c 
wres%u �%v%v H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev%t �%w%w H 4 0 2 3 2 5 0 3 - 9 F B 1 - 4 7 9 3 - A A 3 0 - 7 3 F 5 9 7 3 8 3 3 8 4
�d kfrmID  � %x%x %y�a%z�`%y b�_%{�^
�_ 
wres%{ �%|%| H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev%z �%}%} H A D F 5 E 7 0 B - 5 7 9 E - 4 A 6 1 - B C 5 4 - B 6 0 0 E 2 A F 1 6 F F
�` kfrmID  � %~%~ %�]%��\% b�[%��Z
�[ 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev%� �%�%� H B F 1 E 8 B 1 2 - 9 2 C 4 - 4 E 9 B - 9 E 4 E - D 0 0 D 1 B F A 7 B 4 6
�\ kfrmID  � %�%� %��Y%��X%� b�W%��V
�W 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev%� �%�%� H F 6 3 C 8 8 1 C - 7 B 8 C - 4 9 A 5 - 9 F 4 9 - E 1 3 9 D 2 1 5 D 8 7 9
�X kfrmID  � %�%� %��U%��T%� b�S%��R
�S 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev%� �%�%� H F 7 5 F 6 8 1 F - F 8 D 7 - 4 E 6 B - 9 9 9 6 - 9 2 9 1 5 1 D D C 4 E D
�T kfrmID  � %�%� %��Q%��P%� b�O%��N
�O 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev%� �%�%� H F A 6 3 4 2 5 4 - 9 4 E 7 - 4 B 0 D - A 1 9 2 - 4 B A B 5 C C 9 D 1 E 6
�P kfrmID  � %�%� %��M%��L%� b�K%��J
�K 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev%� �%�%� H 5 4 5 B 7 9 1 D - 3 C 5 8 - 4 A 8 0 - 8 0 C A - E B 3 C 8 B 0 6 D 8 B 0
�L kfrmID  � %�%� %��I%��H%� b�G%��F
�G 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev%� �%�%� H 5 5 B 3 0 8 D E - B A 1 C - 4 B 7 F - B 7 A 7 - 9 0 4 B 7 7 5 7 E 6 7 E
�H kfrmID  � %�%� %��E%��D%� b�C%��B
�C 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev%� �%�%� H 4 D C 4 8 6 F 2 - C 6 2 4 - 4 B C C - A 5 5 A - F 2 6 5 6 9 1 F 3 A D 2
�D kfrmID  � %�%� %��A%��@%� b�?%��>
�? 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev%� �%�%� H D 8 3 5 4 5 9 E - 9 D 6 1 - 4 C 1 2 - 9 C 7 D - 6 C 1 0 1 4 C F C 3 F 2
�@ kfrmID  � %�%� %��=%��<%� b�;%��:
�; 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev%� �%�%� H B E 0 B 5 0 B 2 - 3 0 C E - 4 4 0 3 - 8 F 8 F - 3 C 9 5 1 A 7 D 8 D C 4
�< kfrmID  � %�%� %��9%��8%� b�7%��6
�7 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev%� �%�%� H 4 F 3 C D D 0 5 - 0 A F 0 - 4 2 1 E - A 9 8 D - C 1 6 C 3 9 C 9 3 F 5 6
�8 kfrmID  � %�%� %��5%��4%� b�3%��2
�3 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�2 kfrmID  
�5 
wrev%� �%�%� H 2 1 A C 2 3 D D - 4 6 3 7 - 4 8 F 5 - A E D F - 4 6 9 4 0 9 B B 6 C 3 5
�4 kfrmID  � %�%� %��1%��0%� b�/%��.
�/ 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�. kfrmID  
�1 
wrev%� �%�%� H D 9 B 3 B A 9 A - 5 B F C - 4 2 3 1 - A 3 3 A - 8 B C 6 C D 2 A 9 E 0 F
�0 kfrmID  � %�%� %��-%��,%� b�+%��*
�+ 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�* kfrmID  
�- 
wrev%� �%�%� H 2 8 C 8 2 1 C F - 6 6 A 4 - 4 4 2 2 - 8 1 9 9 - 9 5 9 A 7 5 B 0 B A 2 1
�, kfrmID  � %�%� %��)%��(%� b�'%��&
�' 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�& kfrmID  
�) 
wrev%� �%�%� H D E D 6 1 A 0 1 - 6 9 3 0 - 4 2 A 6 - A F A 7 - 8 F 8 7 0 C 0 9 7 F 2 6
�( kfrmID  � %�%� %��%%��$%� b�#%��"
�# 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�" kfrmID  
�% 
wrev%� �%�%� H 5 2 5 B F 6 B C - 8 B E 4 - 4 E 3 4 - A D 6 2 - 9 8 7 8 C 8 0 E 8 C 5 D
�$ kfrmID  � %�%� %��!%�� %� b�%��
� 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�! 
wrev%� �%�%� H 9 C 5 2 4 9 7 B - D 6 F B - 4 9 F B - 8 F B F - A 8 6 8 9 9 C 4 4 B 8 6
�  kfrmID  � %�%� %��%��%� b�%��
� 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%� �%�%� H 4 C 7 6 3 F 1 E - A 0 0 1 - 4 5 F F - B 7 5 8 - E 9 D A E 1 7 A B A 8 1
� kfrmID  � %�%� %��%��%� b�%��
� 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%� �%�%� H C 8 7 D 1 7 2 2 - 7 0 7 3 - 4 1 2 9 - A 7 2 1 - B 0 1 2 C F 5 2 9 B C F
� kfrmID  � %�%� %��%��%� b�%��
� 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%� �%�%� H A A D 0 D 9 5 A - 5 7 4 0 - 4 B 3 0 - B 8 E 6 - 7 1 D 8 1 C 7 9 F 8 5 5
� kfrmID  � %�%� %��%��%� b�%��
� 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev%� �%�%� H 9 6 F A 4 4 F 1 - 9 7 6 B - 4 1 C F - 8 B 5 E - 4 3 7 E A 5 4 E 7 6 F 9
� kfrmID  � %�%� %��%��%� b�%��

� 
wres%� �%�%� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�
 kfrmID  
� 
wrev%� �%�%� H 1 1 2 2 3 A C 1 - 9 C 7 6 - 4 8 F 9 - 8 B 3 2 - 9 0 E 6 B D 0 2 A 2 8 B
� kfrmID  � %�%� %��	%��%� b�%��
� 
wres%� �& &  H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�	 
wrev%� �&& H 8 4 D 2 A E F 0 - 3 D 3 F - 4 F A 9 - 9 3 0 7 - 0 5 F 2 3 D 7 8 9 C 0 6
� kfrmID  � && &�&�& b�&�
� 
wres& �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev& �&& H B B 3 C A 3 9 1 - C 6 4 7 - 4 3 D 8 - 9 7 2 5 - 4 3 F B 5 8 7 1 D A B 5
� kfrmID  � && &	�&
� &	 b��&��
�� 
wres& �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
� 
wrev&
 �&& H 7 6 7 C 0 5 2 7 - B 3 4 C - 4 1 0 8 - 9 C A 7 - 1 9 7 5 7 4 E B 6 E 6 A
�  kfrmID  � && &��&��& b��&��
�� 
wres& �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �&& H 6 D 0 0 2 0 B E - 5 1 2 3 - 4 C 8 0 - 8 A 4 D - 3 C E A 0 5 4 B 5 A 9 C
�� kfrmID  � && &��&��& b��&��
�� 
wres& �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �&& H 3 E 0 8 F 9 8 F - C F 1 6 - 4 4 C E - 9 4 5 6 - 1 E B D 4 1 3 A 8 A 4 0
�� kfrmID  � && &��&��& b��&��
�� 
wres& �&& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev& �&& H B 1 4 6 9 5 A 7 - 0 D 5 9 - 4 0 A A - 8 9 F 1 - 9 C 9 0 C 3 E 6 6 3 1 F
�� kfrmID  � & &  &!��&"��&! b��&#��
�� 
wres&# �&$&$ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&" �&%&% H 3 7 A 4 F 5 E 5 - A C B 4 - 4 8 9 3 - B 2 E F - 1 F 7 3 0 5 2 B 1 3 8 6
�� kfrmID  � &&&& &'��&(��&' b��&)��
�� 
wres&) �&*&* H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&( �&+&+ H 7 5 E 7 D 9 A 2 - 3 B 5 E - 4 7 6 9 - 8 3 C B - E A 9 C 3 4 5 5 8 0 E 2
�� kfrmID  � &,&, &-��&.��&- b��&/��
�� 
wres&/ �&0&0 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&. �&1&1 H 2 8 4 0 4 C 1 D - A 5 8 8 - 4 F 6 5 - 9 1 3 0 - E C 9 F 4 E C 3 9 0 2 4
�� kfrmID  � &2&2 &3��&4��&3 b��&5��
�� 
wres&5 �&6&6 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&4 �&7&7 H 0 6 7 6 7 A E F - A 8 A A - 4 8 1 4 - A 9 B 2 - 1 0 A 8 A 9 C 8 D 6 9 9
�� kfrmID  � &8&8 &9��&:��&9 b��&;��
�� 
wres&; �&<&< H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&: �&=&= H A F 6 F 1 F 5 9 - 8 7 2 0 - 4 F D 7 - 8 1 0 C - 4 C C 5 A 4 C 9 D C 1 6
�� kfrmID  � &>&> &?��&@��&? b��&A��
�� 
wres&A �&B&B H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&@ �&C&C H 6 2 6 A 1 9 B 1 - 7 4 4 A - 4 2 5 B - A 1 5 8 - 1 F 2 B 0 3 2 5 5 E 0 F
�� kfrmID  � &D&D &E��&F��&E b��&G��
�� 
wres&G �&H&H H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&F �&I&I H 2 C 6 0 0 0 B F - 6 2 B B - 4 2 5 3 - B 1 A D - 2 C D 1 4 F E 2 8 C F A
�� kfrmID  � &J&J &K��&L��&K b��&M��
�� 
wres&M �&N&N H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&L �&O&O H 7 5 F 6 0 B A 5 - 5 4 7 2 - 4 3 2 D - 9 D C 6 - 1 2 2 E 5 5 2 1 C 4 9 8
�� kfrmID  � &P&P &Q��&R��&Q b��&S��
�� 
wres&S �&T&T H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&R �&U&U H 7 B A C C 5 0 4 - 1 B 8 1 - 4 0 7 5 - 8 F 1 8 - 5 A 9 A 8 6 E 5 E C A 5
�� kfrmID  � &V&V &W��&X��&W b��&Y��
�� 
wres&Y �&Z&Z H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&X �&[&[ H 1 8 9 F 6 2 F 7 - 8 0 9 8 - 4 3 8 5 - 8 5 4 4 - C 8 2 2 F 6 2 D 4 B 7 A
�� kfrmID  � &\&\ &]��&^��&] b��&_��
�� 
wres&_ �&`&` H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&^ �&a&a H A F D 7 8 4 B 8 - 9 1 1 8 - 4 B 2 B - 9 D D 9 - 5 6 B 5 C F 0 9 A B E 4
�� kfrmID  � &b&b &c��&d��&c b��&e��
�� 
wres&e �&f&f H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�� kfrmID  
�� 
wrev&d �&g&g H D 7 8 D D 0 6 6 - 3 3 F B - 4 E 9 B - A D D C - 8 5 5 0 0 2 4 B 2 8 C B
�� kfrmID  � &h&h &i��&j��&i b�&k�
� 
wres&k �&l&l H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
�� 
wrev&j �&m&m H A B 1 C B B E E - 9 C E 9 - 4 3 8 8 - B 8 8 A - 2 D 7 7 D E 9 8 8 C 3 0
�� kfrmID  � &n&n &o�&p�&o b�&q�
� 
wres&q �&r&r H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&p �&s&s H F 1 6 E 6 9 B D - 2 0 1 1 - 4 5 5 C - B 4 C B - 1 A D 8 E 2 0 2 1 A F E
� kfrmID  � &t&t &u�&v�&u b�&w�
� 
wres&w �&x&x H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&v �&y&y H F 4 C D D 1 4 A - A A 6 B - 4 2 8 0 - 8 9 6 7 - 5 0 4 7 D 0 E 4 5 4 A C
� kfrmID  � &z&z &{�&|�&{ b�&}�
� 
wres&} �&~&~ H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&| �&& H A C E A 5 9 0 A - F 8 3 D - 4 5 B 5 - A 8 B B - 5 6 0 B 3 9 B B A E 5 E
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 5 2 E 3 1 B 4 3 - 1 D 0 B - 4 C B 4 - 8 9 6 7 - 5 C 7 1 6 3 5 5 D D F A
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 2 7 E 1 D A 9 D - 8 9 1 D - 4 C 1 5 - 8 A 5 B - C 8 3 F 6 C 5 B A 6 4 7
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H A 0 7 D 1 F 2 5 - 9 7 F 7 - 4 B C D - 8 A F 6 - 9 9 5 B 8 6 8 F 2 1 0 9
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 7 8 6 D D 2 0 4 - 5 F 8 B - 4 2 0 F - B B 9 B - 3 B 9 C B C 5 5 F 9 E F
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 8 B 7 8 6 3 0 E - 7 C 4 E - 4 A 7 B - B A F A - 1 B 1 9 1 A 5 F 2 A B 1
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 1 0 1 D 4 6 9 2 - D 9 0 E - 4 9 7 4 - A B 0 E - A 1 7 F D 9 8 8 9 D 2 2
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 9 7 B 3 2 0 E 6 - C 9 9 F - 4 C 6 2 - B 2 E C - 8 5 B 9 8 2 1 6 5 4 2 1
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H E 2 F 5 D 3 E 6 - 2 1 7 8 - 4 F B A - A B E E - F E 1 A 5 C E 7 5 0 4 C
� kfrmID  � &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H 5 0 4 A E 0 6 6 - 8 8 4 3 - 4 E 9 A - 9 F 0 9 - 5 7 A 4 1 A D 7 8 8 8 7
� kfrmID  	  &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H E 9 3 4 3 8 C 8 - D C 5 A - 4 1 E C - 8 7 4 3 - 4 F 3 3 A 9 9 1 5 3 A 7
� kfrmID  	 &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H D B 5 E 8 C 3 6 - D 7 C 8 - 4 2 2 1 - B 2 3 2 - 9 0 7 0 4 3 D A 8 B B E
� kfrmID  	 &�&� &��&��&� b�&��
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
� kfrmID  
� 
wrev&� �&�&� H C C D 4 3 8 F B - C 5 7 D - 4 5 8 9 - B 4 F 6 - 6 8 A 8 6 7 5 6 B C 8 8
� kfrmID  	 &�&� &��&��&� b�&��~
� 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�~ kfrmID  
� 
wrev&� �&�&� H 3 B 9 5 9 F 2 A - A 8 2 1 - 4 E E A - 8 4 2 F - 2 9 B 1 9 B 6 9 A 5 A 3
� kfrmID  	 &�&� &��}&��|&� b�{&��z
�{ 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�z kfrmID  
�} 
wrev&� �&�&� H B 2 D 7 F 5 4 C - D 7 0 2 - 4 9 5 3 - 9 E 8 5 - 6 F F 4 C 5 A 7 4 4 C 1
�| kfrmID  	 &�&� &��y&��x&� b�w&��v
�w 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�v kfrmID  
�y 
wrev&� �&�&� H 4 4 9 E B 8 3 8 - 6 E C F - 4 1 7 8 - 9 B 9 1 - 9 8 D D D A D B 9 7 4 5
�x kfrmID  	 &�&� &��u&��t&� b�s&��r
�s 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�r kfrmID  
�u 
wrev&� �&�&� H D 3 4 2 B 7 D 2 - E 2 6 E - 4 C 8 1 - 8 C 4 6 - 0 9 B 1 3 6 F 7 0 5 A 5
�t kfrmID  	 &�&� &��q&��p&� b�o&��n
�o 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�n kfrmID  
�q 
wrev&� �&�&� H A D 9 B 5 8 8 0 - 6 0 0 8 - 4 1 F A - 9 8 7 A - 3 C 0 8 A 0 5 8 D 4 B 7
�p kfrmID  	 &�&� &��m&��l&� b�k&��j
�k 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�j kfrmID  
�m 
wrev&� �&�&� H 9 9 7 E F 9 B 7 - B 7 1 A - 4 5 B 5 - 9 C 7 8 - D B 2 2 F 8 3 E 0 6 7 5
�l kfrmID  		 &�&� &��i&��h&� b�g&��f
�g 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�f kfrmID  
�i 
wrev&� �&�&� H 5 7 2 A 2 0 B 9 - A 7 6 6 - 4 7 8 C - B 6 4 D - 2 F A 0 B A F E 3 5 E 6
�h kfrmID  	
 &�&� &��e&��d&� b�c&��b
�c 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�b kfrmID  
�e 
wrev&� �&�&� H 5 F 3 C E D 3 C - F C 9 6 - 4 F 3 8 - 9 1 3 B - 2 D 3 2 1 2 4 D F E 5 9
�d kfrmID  	 &�&� &��a&��`&� b�_&��^
�_ 
wres&� �&�&� H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�^ kfrmID  
�a 
wrev&� �&�&� H A 1 B 9 F 1 0 2 - 6 6 6 1 - 4 6 9 2 - 8 F C B - B E 6 D 5 8 4 7 6 D A 2
�` kfrmID  	 &�&� &��]' �\&� b�['�Z
�[ 
wres' �'' H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�Z kfrmID  
�] 
wrev'  �'' H A 5 9 2 4 8 5 8 - 6 7 9 B - 4 2 3 4 - 9 1 4 E - 6 3 6 D 9 A C C 0 0 D 9
�\ kfrmID  	 '' '�Y'�X' b�W'�V
�W 
wres' �'' H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�V kfrmID  
�Y 
wrev' �'	'	 H D 9 1 7 E 3 E 1 - F 4 A 8 - 4 7 8 B - B 6 1 7 - 5 9 F 2 A C 6 2 9 5 5 E
�X kfrmID  	 '
'
 '�U'�T' b�S'�R
�S 
wres' �'' H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�R kfrmID  
�U 
wrev' �'' H 6 7 3 5 0 5 6 B - C 8 D 4 - 4 7 C 7 - 8 4 9 7 - B 0 B D 9 5 E D 4 E 9 2
�T kfrmID  	 '' '�Q'�P' b�O'�N
�O 
wres' �'' H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�N kfrmID  
�Q 
wrev' �'' H 9 E D 6 A 4 8 0 - F C 8 8 - 4 B E A - 9 8 1 7 - 8 5 B 7 8 C 8 7 4 3 A F
�P kfrmID  	 '' '�M'�L' b�K'�J
�K 
wres' �'' H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�J kfrmID  
�M 
wrev' �'' H 8 D D F 9 6 9 B - 9 0 E B - 4 3 7 8 - 8 6 1 E - E 0 3 2 4 8 7 F F 1 1 5
�L kfrmID  	 '' '�I'�H' b�G'�F
�G 
wres' �' '  H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�F kfrmID  
�I 
wrev' �'!'! H 6 C C 5 8 0 C F - 8 E 5 8 - 4 0 7 A - 9 F D 5 - 3 C 9 A 2 3 5 0 A 4 2 F
�H kfrmID  	 '"'" '#�E'$�D'# b�C'%�B
�C 
wres'% �'&'& H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�B kfrmID  
�E 
wrev'$ �'''' H C C 5 1 F 2 4 9 - E 7 8 6 - 4 9 5 A - 9 2 3 3 - D 5 E 8 4 7 A 1 7 1 6 6
�D kfrmID  	 '('( ')�A'*�@') b�?'+�>
�? 
wres'+ �',', H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�> kfrmID  
�A 
wrev'* �'-'- H E B 3 A 6 A C 8 - 8 6 E 0 - 4 9 2 D - 9 5 A C - C 4 9 7 5 A 6 3 2 2 7 A
�@ kfrmID  	 '.'. '/�='0�<'/ b�;'1�:
�; 
wres'1 �'2'2 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�: kfrmID  
�= 
wrev'0 �'3'3 H 0 1 0 0 8 D 3 8 - 0 8 F 3 - 4 D B 1 - A F C 0 - 7 A 5 C 6 2 4 A 8 2 9 A
�< kfrmID  	 '4'4 '5�9'6�8'5 b�7'7�6
�7 
wres'7 �'8'8 H 3 3 D 5 6 6 C D - 5 F D 3 - 4 3 E 8 - A D E 8 - 3 A 6 3 8 6 8 D 5 9 5 B
�6 kfrmID  
�9 
wrev'6 �'9'9 H A 7 A F F 1 8 C - 9 9 B 1 - 4 1 1 1 - 9 C 4 5 - B E 6 6 6 C C 1 0 0 F D
�8 kfrmID  � ldt     �)�p� ldt     �)��� ldt     ���
�  boovfals� �':': H 0 4 D D 0 2 4 5 - 7 6 0 F - 4 0 B 7 - 8 7 9 D - 7 1 F 3 9 1 A 2 0 F 7 2� �';'; " W A   W O L V E S   &   H O U R S� �'<'<  m i s s i n g   v a l u e� �'='=  m i s s i n g   v a l u e� �'>'>  m i s s i n g   v a l u e� �'?'?  m i s s i n g   v a l u e
�� boovtrue
�� ****E4no� ldt     �|�H�� � �5'@�5  '@   � �'A'A  m i s s i n g   v a l u e� �'B'B  m i s s i n g   v a l u e� �'C'C * S a n g y e   I n c e - J o h a n n s e n� �'D'D . s a n g y e i j @ w e s t e r n l a w . o r g� �'E'E  a c c e p t e d� �'F'F * S a n g y e   I n c e - J o h a n n s e n� �'G'G . s a n g y e i j @ w e s t e r n l a w . o r g��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��   ascr  ��ޭ