<template>
  <v-app>
    <!-- NavBar left -->
    <v-navigation-drawer v-model="drawer" :clipped="$vuetify.breakpoint.lgAndUp" app>
      <v-list dense>
        <template v-for="item in items">
          <v-list-item :key="item.text" link>
            <v-list-item-action>
              <v-icon>{{ item.icon }}</v-icon>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>
                {{ item.text }}
              </v-list-item-title>
            </v-list-item-content>
          </v-list-item>
        </template>
      </v-list>
    </v-navigation-drawer>
    <!-- NavBar top -->
    <v-app-bar :clipped-left="$vuetify.breakpoint.lgAndUp" app color="blue darken-4" dark>
      <v-app-bar-nav-icon @click.stop="drawer = !drawer" />
      <v-toolbar-title style="width: 300px" class="ml-0 pl-1">
        <span>Application Machine Learning</span>
      </v-toolbar-title>
      <v-spacer />
    </v-app-bar>
    <!-- Content -->
    <v-content>
      <router-view></router-view>
    </v-content>
    <!-- Navbar bottom -->
    <NavBottom v-if="true"></NavBottom>
  </v-app>
</template>

<script>
import NavBottom from '@/components/NavBottom.vue'

export default {
  components: {
    NavBottom
  },
  data: () => ({
    drawer: null,
    items: [
      { icon: 'mdi-contacts', text: 'Contacts' }
    ]
  })
}
</script>
