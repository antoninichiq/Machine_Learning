imagem = cv2.imread(nome_do_arquivo)

# Convertendo para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
imagem_com_filtro = cv2.GaussianBlur(imagem_cinza,(5,5),0) #aplicando filtro para ruído

#Mostrando a imagem desfocada
print(cv2_imshow(imagem_com_filtro))

#Calculando os filtros Canny com diferentes limiares (para testar)
largo = cv2.Canny(imagem_com_filtro,50,220) #threshold baixo = 50 e alto = 220
medio = cv2.Canny(imagem_com_filtro,70,140)
apertado = cv2.Canny(imagem_com_filtro,210,220)

#Mostrando os Mapas de Bordas (Map Edges) com diferentes limiares
print(cv2_imshow(largo))
print(cv2_imshow(medio))
print(cv2_imshow(apertado))
