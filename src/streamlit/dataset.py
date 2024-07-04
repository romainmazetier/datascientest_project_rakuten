import streamlit as st


def display():
    
    st.header("Le Jeu de données")
    st.write("Cette section décrit les différents jeux de données utilisés dans notre application.")
    # Section 1: Explications des données
    st.subheader('1. Les datasets', divider='rainbow')
    st.write("Nous avions en tout 3 jeux de données distincts :  X_train, X_test, y_train")
    st.write("Vu que nous n'avions pas de y_test, nous avons décidé de ne pas utiliser X_test et de diviser X_train et y_train pour créer un nouveau X_test et y_test ")
    st.write("Voici le format de X_train comprenant 84 916 produits: ")
    st.image('image/x_train.png', caption='Premier dataset X_train')
    st.write("Voici le format de y_train : ")
    st.image('image/y_train.png', caption='Second dataset y_train')
    dict_classe = {10:'Livres en langue étrangère, livres d’occasion',
                    40:'Jeux vidéos',
                    50:'Accessoires jeux vidéos',
                    60:'Consoles de jeux vidéos',
                    1140:'Figurines (décorations)',
                    1160:'Cartes à collectionner',
                    1180:'Figurines (jeux de rôle)',
                    1280:'Jeux et jouets',
                    1281:'Jeux de société',
                    1300:'Drones et maquettes',
                    1301:'Baby foot et chaussures de bébé',
                    1302:'Loisirs en extérieur',
                    1320:'Bébé',
                    1560:'Mobilier',
                    1920:'Literie',
                    1940:'Alimentation',
                    2060:'Décorations',
                    2220:'Accessoires pour animaux',
                    2280:'Journaux, magazines, comics',
                    2403:'Mangas, revues, littérature jeunesse',
                    2462:'Jeux vidéos d’occasion',
                    2522:'Papeterie',
                    2582:'Jardin',
                    2583:'Piscines',
                    2585:'Bricolage',
                    2705:'Littérature adulte, éducation, documentaires',
                    2905:'Jeux vidéos dématérialisés'}
    st.dataframe(dict_classe, column_config={"value": "Class Name"})
    st.subheader('2. Les images', divider='rainbow')
    st.write("Comme nous pouvons le voir dans X_train il y a un champ imageid et un champ productid qui vont nous permettre de relier cette donnée à une image, prenons l'exemple du produit n°3, voici son image correspondante :")
    st.image('image/image_457047496_product_50418756.jpg', caption='Image de Donald')