import streamlit as st
import cv2

def display_intro():

    st.title('Projet Rakuten')

    st.header("Introduction")

    st.write("Présentation du challenge : https://challengedata.ens.fr/challenges/35")

    col1, col2 = st.columns([0.6,  0.12])
    with col1:
        st.image("image/img_intro_streamlit.png", caption='Aperçu du site Rakuten')

    st.subheader("Le projet", divider='rainbow')
    st.write("Notre projet s’inscrit dans le cadre d’un challenge lancé par Rakuten, qui est l’un des plus importants sites de vente en ligne dans le monde. Le challenge consiste à développer un modèle capable de prédire au mieux la catégorie des produits vendus dans le catalogue français de Rakuten, lesquels sont répartis en 27 catégories. On dispose pour chacun d’une désignation textuelle et d’une image, et éventuellement d’une description. Il s’agit donc d’un problème de classification supervisée multi-classes et multimodale.\n\n\
Dans sa description du challenge, Rakuten explique que la classification de produits manuelle ou basée sur des règles prédéfinies n’est pas une solution adaptée pour les sites de vente en ligne lorsque les catégories de produits sont aussi nombreuses (27 catégories). Mais Rakuten n’explique pas pour autant quel est exactement son intérêt à lancer ce challenge. L’analyse du jeu de données permet de comprendre que la catégorisation des produits est en fait déjà produite par une IA : on sait donc que le challenge ne vise pas à faire développer un modèle qui serve à remplacer une classification manuelle des produits.\n\n\
Quel que soit l’objectif de Rakuten dans ce challenge, pour nous, ce projet a été d’un grand intérêt pédagogique, puisqu’il nous a permis d’aborder aussi bien la computer vision que le traitement du langage naturel, et nous a même donné l’occasion d’apprendre à construire des modèles multimodaux.")



def display_conclu():



    st.header("Conclusion")
    st.subheader("Les difficultés rencontrées", divider='rainbow')
    col1, col2 = st.columns([0.05, 0.95])
    with col2:
        st.markdown('<b>Partage des modèles &mdash; diversité des environnements<b>', unsafe_allow_html=True)
    st.write("Nous avons travaillé sur différents OS (Windows, Mac, et Linux), et pour certains sur nos ordinateurs personnels, pour d’autres sur des machines virtuelles distantes à cause de contraintes matérielles. Ceci nous a conduit à utiliser des versions différentes de nos librairies. Par conséquent, nous avons rencontré des problèmes au moment de se partager les modèles que nous avions entraînés, parce qu’un modèle entraîné sous une certaine version de keras, par exemple, ne fonctionne pas forcément sous une autre. Uniformiser nos environnements dans la mesure du possible et adapter notre code en conséquence nous a pris du temps.")

    col1, col2 = st.columns([0.05, 0.95])
    with col2:
        st.markdown('<b>Contraintes matérielles<b>', unsafe_allow_html=True)
    st.write("L'entraînement des modèles était parfois très exigeant en puissance de calcul et en mémoire. Par exemple, ceux d'entre nous qui s'étaient chargés d'essayer de mettre en oeuvre de l'oversampling sur les images n'y sont pas parvenus par manque de mémoire.")

    st.subheader("Pistes d'amélioration", divider='rainbow')
    st.markdown("""
Quelques pistes d’amélioration que nous avons identifiées et que nous aurions pu suivre si nous avions disposé de plus de temps : 
- Utiliser la variable indicatrice de l’absence de description, créée au début de notre projet, et dont on a montré dans notre exploration des données qu’elle entretenait une corrélation avec la catégorie du produit.
- Poursuivre le travail tout juste commencé sur les modèles bimodaux. C’est la piste que nous avons commencé à explorer le plus tardivement, et donc celle où nous sommes allés le moins loin. Mais on peut aussi penser qu’il s’agit de la voie avec le meilleur potentiel. 
""")